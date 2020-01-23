######################################################################################
# Evaluates regressor from loaded model
# Usage:
# python3 isotrackApplyRegressor.py -PU root://cmseos.fnal.gov//store/user/sghosh/ISOTRACK/DIPI_2021_PUpart.root -M ./models/model1.h5 -O corrfac_regression.txt
######################################################################################
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras import regularizers
from sklearn.metrics import roc_curve, auc
from keras.layers import Activation
from keras import backend as K
from keras.models import load_model
import uproot


# In[2]:

parser = argparse.ArgumentParser()
parser.add_argument("-PU", "--filePU",help="input PU file",default="root://cmseos.fnal.gov//store/user/sghosh/ISOTRACK/DIPI_2021_PUpart.root")

parser.add_argument("-M", "--modelname",help="model file name",default="./models/model.h5")
parser.add_argument("-O", "--opfilename",help="output text file name",default="corrfac_regression.txt")


fName1 = parser.parse_args().filePU
modeln = parser.parse_args().modelname
foutput = parser.parse_args().opfilename



#fName1='/eos/uscms/store/user/sghosh/ISOTRACK/DIPI_2021_PUpart.root'
tree1 = uproot.open(fName1,xrootdsource=dict(chunkbytes=1024**3, limitbytes=1024**3))['HcalIsoTrkAnalyzer/CalibTree']
print ("loaded files")

branchespu = ['t_Run','t_Event','t_nVtx','t_ieta','t_iphi','t_p','t_pt','t_gentrackP','t_eMipDR','t_eHcal','t_eHcal10','t_eHcal30','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh']
dictpu = tree1.arrays(branches=branchespu)
dfspu = pd.DataFrame.from_dict(dictpu)
dfspu.columns=branchespu
print ("sample size:",dfspu.shape[0])


# In[3]:


dfspu['t_delta']=dfspu['t_eHcal30']-dfspu['t_eHcal10']
keepvars =['t_nVtx', 't_ieta', 't_eHcal10', 't_eHcal30', 't_delta', 't_hmaxNearP','t_emaxNearP', 't_hAnnular', 't_eAnnular', 't_rhoh', 't_pt','t_eHcal', 't_p', 't_eMipDR']

df = dfspu[keepvars].copy()
df['t_eHcal_xun'] = df['t_eHcal']
df['t_delta_un'] = df['t_delta']
df['t_ieta_un'] = df['t_ieta']

#cols_to_stand = ['t_nVtx','t_ieta','t_eHcal10', 't_eHcal30','t_rhoh','t_eHcal_x']
#cols_to_minmax = ['t_delta', 't_hmaxNearP','t_emaxNearP', 't_hAnnular', 't_eAnnular','t_pt']
cols_to_minmax = ['t_delta', 't_hmaxNearP','t_emaxNearP', 't_hAnnular', 't_eAnnular','t_pt','t_nVtx','t_ieta','t_eHcal10', 't_eHcal30','t_rhoh','t_eHcal']
#df[cols_to_stand] = df[cols_to_stand].apply(lambda x: (x - x.mean()) /(x.std()))
#df[cols_to_minmax] = df[cols_to_minmax].apply(lambda x: (x - x.mean()) /  (x.max() - x.min()))
#                                            #(x.max() - x.min()))
df[cols_to_minmax] = df[cols_to_minmax].apply(lambda x: (x - x.min()) /  (x.max() - x.min()))


uncorrected_values = df['t_eHcal_xun'].values
print (uncorrected_values.shape)
print ('data vars:',df.keys())
data = df.values
X_train = data[:,0:12]


# In[4]:


model = load_model(modeln)
preds = model.predict(X_train,verbose=1)
preds = preds.reshape(preds.shape[0])
print (preds.shape)


# In[5]:


plt.hist(preds, bins =100, range=(0,100),label='predicted enerhy',alpha=0.6)
plt.savefig('predicted_Edist.png')


# In[6]:


eventnumarray = np.arange(0,X_train.shape[0],1,dtype=int)
eventnumarray = eventnumarray.reshape(eventnumarray.shape[0],1)
corrfac = preds/uncorrected_values
corrfac = corrfac.reshape(corrfac.shape[0],1)
fileo = np.hstack((eventnumarray,corrfac))


# In[ ]:


np.savetxt(foutput, fileo)   

