######################################################################################
# Evaluates regressor from loaded model
# Usage:
# source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc8-opt/setup.csh
# python3 isotrackApplyRegressor.py -PU root://eoscms.cern.ch//eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/SinglePion_E-50_Eta-0to3_Run3Winter21_112X_PU.root -M ./models/model1.h5 -B endcap -O corrfac1.txt
# python3 isotrackApplyRegressor.py -PU root://eoscms.cern.ch//eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/SinglePion_E-50_Eta-0to3_Run3Winter21_112X_PU.root -M ./models/model2.h5 -B barrel -O corrfac2.txt
######################################################################################
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import uproot


# In[2]:

parser = argparse.ArgumentParser()
parser.add_argument("-PU", "--filePU",help="input PU file",default="root://eoscms.cern.ch//eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/SinglePion_E-50_Eta-0to3_Run3Winter21_112X_PU.root")
parser.add_argument("-B", "--ifbarrel",help="barrel / endcap",default='barrel')
parser.add_argument("-M", "--modelname",help="model file name",default="./models/model.h5")
parser.add_argument("-O", "--opfilename",help="output text file name",default="corrfac_regression.txt")


fName1 = parser.parse_args().filePU
modeln = parser.parse_args().modelname
foutput = parser.parse_args().opfilename
barrelflag = parser.parse_args().ifbarrel



#fName1='/eos/uscms/store/user/sghosh/ISOTRACK/DIPI_2021_PUpart.root'
tree1 = uproot.open(fName1,xrootdsource=dict(chunkbytes=1024**3, limitbytes=1024**3))['HcalIsoTrkAnalyzer/CalibTree']
print ("loaded files")

branchespu = ['t_Run','t_Event','t_nVtx','t_ieta','t_iphi','t_p','t_pt','t_gentrackP','t_eMipDR','t_eHcal','t_eHcal10','t_eHcal30','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh']
dictpu = tree1.arrays(branches=branchespu)
#dictpu = tree1.arrays(branches=branchespu,entrystart=0, entrystop=100000)
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

#### if using global norm
#cols_to_minmax = ['t_delta', 't_hmaxNearP','t_emaxNearP', 't_hAnnular', 't_eAnnular','t_pt','t_nVtx','t_ieta','t_eHcal10', 't_eHcal30','t_rhoh','t_eHcal']
#df[cols_to_minmax] = df[cols_to_minmax].apply(lambda x: (x - x.min()) /  (x.max() - x.min())  if (x.max() - x.min() > 0) else 1.0/200.0)

#### if using training norm
if barrelflag=='barrel':
    var = ['t_nVtx', 't_ieta', 't_eHcal10', 't_eHcal30', 't_delta', 't_hmaxNearP', 't_emaxNearP', 't_hAnnular', 't_eAnnular', 't_rhoh', 't_pt', 't_eHcal_x', 't_eHcal_y', 't_p', 't_eMipDR', 't_eHcal_xun', 't_delta_un', 't_ieta_un']
    mina = [20, -15, 17.33086721925065, 17.655660001095384, 0.0, -1.0, -1.0, 0.0, -5.33359869197011, 4.093925265397289, 20.783629520782718, 16.998163268435746, 20.000221125315875, 40.074083721419896, 0.0, 16.998163268435746, 0.0, -15]
    maxa = [138, 15, 138.26640254695667, 155.83508832909865, 50.9643486259738, 17.140547914961605, 2870.424876287056, 35.727171580074355, 17.763740802183747, 26.38359781195008, 59.169594172331905, 133.07561272289604, 122.8542027361691, 59.977312414583295, 0.9999987781047821, 133.07561272289604, 50.9643486259738, 15]
else:
    var = ['t_nVtx', 't_ieta', 't_eHcal10', 't_eHcal30', 't_delta', 't_hmaxNearP', 't_emaxNearP', 't_hAnnular', 't_eAnnular', 't_rhoh', 't_pt', 't_eHcal_x', 't_eHcal_y', 't_p', 't_eMipDR', 't_eHcal_xun', 't_delta_un', 't_ieta_un']
    mina = [20, -27, 20.934556022286415, 27.213239994598553, 0.7803188574616797, -1.0, -1.0, 0.0, -57.422806948423386, 4.435386319143829, 6.013227444563335, 18.5278014652431, 20.00695458613336, 40.00339106433163, 0.0, 18.5278014652431, 0.7803188574616797, -27]
    maxa = [117, 28, 584.1150933708996, 822.056631873711, 424.1259534251876, 19.999632518345965, 13218.009997109137, 185.96148682758212, 128.60208600759506, 25.911101538538524, 28.928444814127907, 428.99590471945703, 111.25579111767001, 59.98427126709689, 0.9999948740005493, 428.99590471945703, 424.1259534251876, 28]

print(mina)
print(maxa)

normkey = ['t_nVtx', 't_ieta', 't_eHcal10', 't_eHcal30', 't_delta', 't_hmaxNearP', 't_emaxNearP', 't_hAnnular', 't_eAnnular', 't_rhoh', 't_pt', 't_eHcal']
k = 0
for i in df.keys():
    if i not in normkey:
        break
    #print(normkey[k],mina[k],maxa[k])
    df[i]=abs(df[i]-mina[k])/(maxa[k] - mina[k])
    k+=1


df['t_eHcal_xun'] = df['t_eHcal_xun'].apply(lambda x: 1.0 if (x==0) else x)
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
if barrelflag=='barrel':
    tag = 'barrel'
else:
    tag = 'endcap'

print(tag)

plt.hist(preds, bins =100, range=(0,100),label='predicted enerhy',alpha=0.6)
plt.savefig('predicted_Edist_'+tag+'.png')
plt.close()

parray = dfspu['t_p'].values
mipdr = dfspu['t_eMipDR'].values
plt.hist(preds/(parray - mipdr), bins =100, range=(0,10),label='predicted e/p-ecal',alpha=0.6)
plt.savefig('predicted_eopdist_'+tag+'.png')
plt.close()



# In[6]:
eventnumarray = np.arange(0,X_train.shape[0],1,dtype=int)
eventnumarray = eventnumarray.reshape(eventnumarray.shape[0],1)
#runnumarray = dfspu['t_Run'].values
#runnumarray = runnumarray.reshape(runnumarray.shape[0],1)
#eventnumarray = dfspu['t_Event'].values
#eventnumarray = eventnumarray.reshape(eventnumarray.shape[0],1)
ietaarray = dfspu['t_ieta'].values
ietaarray = ietaarray.reshape(ietaarray.shape[0],1)
#iphiarray = dfspu['t_iphi'].values
#iphiarray = iphiarray.reshape(iphiarray.shape[0],1)
corrfac = preds/uncorrected_values
corrfac = corrfac.reshape(corrfac.shape[0],1)
fileo = np.hstack((eventnumarray,ietaarray,corrfac))
#fileo = np.hstack((runnumarray,eventnumarray,ietaarray,iphiarray,corrfac))
np.savetxt(foutput, fileo,fmt=['%d','%d','%.10f'])   


# In[ ]:




