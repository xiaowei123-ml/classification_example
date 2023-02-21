import os
import sys
import torch.nn
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])
model_path = sys.argv[3]
train_start_date_str = sys.argv[4]
train_end_date_str = sys.argv[5]
test_start_date_str = sys.argv[6]
test_end_date_str = sys.argv[7]


df_train = df[(df['date_time']>=train_start_date_str) & (df['date_time']<=train_end_date_str)]
df_test = df2[(df2['date_time']>=test_start_date_str) & (df2['date_time']<=test_end_date_str)]

xy = df_train[['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','flag','diff']].to_numpy()
xy = xy.astype(np.float32)
x_data=torch.from_numpy(xy[0:-1,0:10])
y_data=torch.Tensor(xy[0:-1,10:11])

xy_test = df_test[['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','flag','diff']].to_numpy()
xy_test = xy_test.astype(np.float32)
x_test_data=torch.from_numpy(xy_test[0:-1,0:10])
y_test_data=torch.Tensor(xy_test[0:-1,10:11])
z_test_data=torch.Tensor(xy_test[0:-1,[-1]])
test_date_time_list = df_test[0:-1]['date_time'].to_list()
 
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1=torch.nn.Linear(10,9)
        self.linear2=torch.nn.Linear(9,8)
        self.linear3=torch.nn.Linear(8,7)
        self.linear4=torch.nn.Linear(7,6)
        self.linear5=torch.nn.Linear(6,5)
        self.linear6=torch.nn.Linear(5,4)
        self.linear7=torch.nn.Linear(4,3)
        self.linear8=torch.nn.Linear(3,2)
        self.linear9=torch.nn.Linear(2,1)
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()
    
    def forward(self, x):
        x=self.tanh(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        x=self.sigmoid(self.linear4(x))
        x=self.sigmoid(self.linear5(x))
        x=self.sigmoid(self.linear6(x))
        x=self.sigmoid(self.linear7(x))
        x=self.sigmoid(self.linear8(x))
        x=self.sigmoid(self.linear9(x))
        return x
    
    def predict(self,x):
        y = self.forward(x)
        return 1 if y > 0.4 else 0
    
    def score(self,x_test,y_test,z_test,input_date_list):
        temp_date_list=['2018-01','2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12','2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12','2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12','2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12','2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12']
        temp_count_list = [0 for x in range(len(temp_date_list))]
        temp_right_count_list = [0 for x in range(len(temp_date_list))]
        temp_total_diff_list = [0.0 for x in range(len(temp_date_list))]
        temp_average_diff_list = [0.0 for x in range(len(temp_date_list))]
        date_index = 0
        
        right_count = 0
        total_diff = 0.0
        plus_right_count=0
        real_right_count=0
        for i in range(len(x_test)):
            result = self.predict(x_test[i])
            if(result==1):
                plus_right_count += 1
                total_diff += z_test[i]
                date_index = temp_date_list.index(input_date_list[i])
                temp_count_list[date_index] += 1
                temp_total_diff_list[date_index] += z_test[i]
                if(y_test[i]==1):
                    real_right_count += 1
                    temp_right_count_list[date_index] += 1
            if result == y_test[i]:
                right_count += 1
        for i in range(len(temp_date_list)):
            if(temp_count_list[i]!=0):
                print(temp_date_list[i],temp_right_count_list[i]/temp_count_list[i],temp_count_list[i],temp_total_diff_list[i]/temp_count_list[i])
        return right_count/len(x_test),total_diff/plus_right_count,real_right_count/plus_right_count

model=Model()

criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.RMSprop(model.parameters(),lr=0.01)
#optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
 
for epoch in range(500000):
    #Forward
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    if(epoch % 2000 ==0):
        print(epoch,loss.item())
 
    #Backward
    optimizer.zero_grad()
    loss.backward()
    #upDatae
    optimizer.step()
PATH = model_path
torch.save(model.state_dict(),PATH)
model.load_state_dict(torch.load(PATH))
win_rate,win_diff,real_win_rate = model.score(x_test_data,y_test_data,z_test_data,test_date_list)
print("test datasets accuracy", win_rate,win_diff,real_win_rate)
