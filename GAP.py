#!/usr/bin/env python
# coding: utf-8

# # steps for data pre-processing

# step1:importing libraries

# tinyurl.com/preprocess-data

# In[52]:


import numpy as np
import matplotlib.pyplot as plt


# step2:importing data set

# In[53]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_bba26c7923b04a17832943e1b244b58e = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='MTHFdZK_I6-W1zWk4b-hEUyJA2ihmrD9b61PU91_ib9B',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_bba26c7923b04a17832943e1b244b58e.get_object(Bucket='graduateadmissionprediction-donotdelete-pr-8bat18q13wgs5s',Key='Admission_Predict_Ver1.1.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()


# In[54]:


dataset


# In[55]:


dataset.isnull().any()


# In[56]:


dataset.drop(['Serial No.'],axis=1,inplace=True)


# In[57]:


dataset


# step4:seperating independent variables and dependent variables in the dataset

# In[58]:


x=dataset.iloc[:,0:7].values


# In[59]:


x


# In[60]:


y=dataset.iloc[:,7:].values


# In[61]:


y


# step6:splitting train and testing data

# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[64]:


x_test


# In[65]:


y_test


# In[66]:


x_train


# In[67]:


y_train


# In[68]:


x.shape


# In[69]:


np.shape(x_test)


# In[70]:


x_train.shape


# # multilinear regression

# In[71]:


plt.scatter(x_train[:,3],y_train)


# In[72]:


from sklearn.linear_model import LinearRegression


# In[73]:


lr=LinearRegression()


# In[74]:


lr.fit(x_train,y_train)


# In[75]:


y_predict=lr.predict(x_test)


# In[76]:


y_predict


# In[77]:


y_predict=lr.predict(x_test)


# In[78]:


from sklearn.metrics import r2_score


# In[79]:


s1=r2_score(y_test,y_predict)
s1


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:












# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[80]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[81]:


wml_credentials={"instance_id": "aba8dfc2-dc08-4645-beab-6e045bf7e870",
  "password": "3a6e1ad4-932d-471e-bf0d-0ec95ddc0dd7",
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "username": "aad11eef-edda-4002-9779-b85b6c05bcdd","acces_key": "Tm1VOfG6W031RM6WY4TDXOmCwv1N0gOqqNShFq7uSzlG"}


# In[82]:


client=WatsonMachineLearningAPIClient(wml_credentials)


# In[83]:


import json
instance_details=client.service_instance.get_details()
print(json.dumps(instance_details,indent=2))


# In[84]:


model_props={client.repository.ModelMetaNames.AUTHOR_NAME:"Sagarika",
             client.repository.ModelMetaNames.AUTHOR_EMAIL:"sagarika@gmail.com",
            client.repository.ModelMetaNames.NAME:"Multi Linear"}


# In[85]:


model_artifact=client.repository.store_model(lr,meta_props=model_props)


# In[86]:


published_model_uid=client.repository.get_model_uid(model_artifact)


# In[87]:


published_model_uid


# In[88]:


created_deployment=client.deployments.create(published_model_uid,name="multilinear")


# In[89]:


scoring_endpoint=client.deployments.get_scoring_url(created_deployment)
scoring_endpoint


# In[90]:


client.deployments.list()


# In[ ]:




