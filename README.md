# Linear-and-Logistic-combined-with-html-report
Reports visualization with html reports

import sweetviz as sv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

from sklearn.model_selection import train_test_split

Data=pd.read_csv("spambase_csv.csv")
print(Data)
print('-'*30)
print(Data.describe())
print('-'*30)
print(Data.shape)
print('-'*30)
print(Data.head())
print('-'*30)
print(Data.tail(10))
print('-'*30)
print(Data.info())
print('-'*30)
print(Data.to_string())
print('-'*30)



X=Data.iloc[:,:-1]
y=Data.iloc[:,:-1]


print(X.head())
print(y.head())

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print("X-train shape",X_train.shape)
print("Y-train shape",y_train.shape)
print('#'*30)
print("X-test shape",X_test.shape)
print("Y-test shape",y_test.shape)

##Prdict the values
Ln=LinearRegression()
Ln.fit(X_train,y_train)

y_pred=Ln.predict(X_test)
print(y_pred)

print(Ln.coef_)
print(Ln.intercept_)

#visulization
plt.scatter(X_test,y_test,c='r')
plt.show()

print("Below is the Logistic Regression Model")

###Logistic Regression
Data=pd.read_csv("spambase_csv.csv")
X=Data.iloc[:,:-1]
y=Data.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
LG=LogisticRegression(max_iter=100)
LG.fit(X_train,y_train)
y_prd=LG.predict(X_test)
print(y_prd)
print("-"*30)
print("Confusion Matrix:\n",confusion_matrix(y_test,y_prd))
print("-"*30)
print("\nClassification Report:\n",classification_report(y_test,y_prd))
print("-"*30)
print("\nAccuracy:",accuracy_score(y_test,y_prd))
print("-"*30)

#Visulizaion
sns.heatmap(confusion_matrix(y_test,y_prd),annot=True)
plt.show()

reports=sv.analyze(Data)
reports.show_html("Example.html")
plt.show()
