######################################################################################
# Trains regressor and saves model for evaluation
# Usage:
# python3 isotrackTrainRegressor.py -I isotk_relval_hi.pkl  -V 1 
# python3 isotrackTrainRegressor.py -I isotk_relval_lo.pkl  -V 2
######################################################################################


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
from tensorflow.keras.models import save_model



parser = argparse.ArgumentParser()
parser.add_argument("-I", "--input",help="input file",default="isotk_relval_hi.pkl")
parser.add_argument("-V", "--modelv",help="model version (any number) ",default="1")



fName = parser.parse_args().input
modv = parser.parse_args().modelv
# In[2]:


df = pd.read_pickle(fName)
print ("vars in file:",df.keys())
print("events in df original:",df.shape[0])
df = df.loc[df['t_eHcal_y'] > 20]
print("events in df after energy cut:",df.shape[0])
df['t_eHcal_xun'] = df['t_eHcal_x']
df['t_delta_un'] = df['t_delta']
df['t_ieta_un'] = df['t_ieta']

mina = []
maxa = []
keya = []


for i in df.keys():
    keya.append(i)
    mina.append(df[i].min())
    maxa.append(df[i].max())

print('var = ',keya)
print('mina = ',mina)
print('maxa = ',maxa)


#cols_to_stand = ['t_nVtx','t_ieta','t_eHcal10', 't_eHcal30','t_rhoh','t_eHcal_x']
#cols_to_minmax = ['t_delta', 't_hmaxNearP','t_emaxNearP', 't_hAnnular', 't_eAnnular','t_pt']
cols_to_minmax = ['t_delta', 't_hmaxNearP','t_emaxNearP', 't_hAnnular', 't_eAnnular','t_pt','t_nVtx','t_ieta','t_eHcal10', 't_eHcal30','t_rhoh','t_eHcal_x']
#df[cols_to_stand] = df[cols_to_stand].apply(lambda x: (x - x.mean()) /(x.std()))
#df[cols_to_minmax] = df[cols_to_minmax].apply(lambda x: (x - x.mean()) /  (x.max() - x.min()))
#                                            #(x.max() - x.min()))
df[cols_to_minmax] = df[cols_to_minmax].apply(lambda x: (x - x.min()) /  (x.max() - x.min()))

data = df.values
print ('data shape:',data.shape)
targets = data[:,12]
targets.shape


# In[3]:


data = df.values
ntest = 20000
testindx = data.shape[0] - ntest
X_train = data[:testindx,0:12]
Y_train = data[:testindx,12]
X_test = data[testindx:,:]
print ("shape of X_train:",X_train.shape)
print ("shape of Y_train:",Y_train.shape)
print ("shape of X_test:",X_test.shape)
meany = np.mean(Y_train)
print ("mean y:",meany)
stdy = np.std(Y_train)
print ("std y:", stdy)


# In[4]:


############################################# marinas correction  form
a0 = [0.973, 0.998,  0.992,  0.965 ]
a1 =  [0,    -0.318, -0.261, -0.406]
a2 = [0,     0,      0.047,  0.089]
def fac0(jeta):
    PU_IETA_1 = 9
    PU_IETA_2 = 16
    PU_IETA_3 = 25
    idx = (int(jeta >= PU_IETA_1) + int(jeta >= PU_IETA_2) + int(jeta >= PU_IETA_3))
    return a0[idx]
def fac1(jeta):
    PU_IETA_1 = 9
    PU_IETA_2 = 16
    PU_IETA_3 = 25
    idx = (int(jeta >= PU_IETA_1) + int(jeta >= PU_IETA_2) + int(jeta >= PU_IETA_3))
    return a1[idx]
def fac2(jeta):
    PU_IETA_1 = 9
    PU_IETA_2 = 16
    PU_IETA_3 = 25
    idx = (int(jeta >= PU_IETA_1) + int(jeta >= PU_IETA_2) + int(jeta >= PU_IETA_3))
    return a2[idx]

vec0 = np.vectorize(fac0)
vec1 = np.vectorize(fac1)
vec2 = np.vectorize(fac2)

X_test[:,17]
eop = (X_test[:,15]/X_test[:,13])
dop = (X_test[:,16]/X_test[:,13])
#mcorrval = vec0(abs(X_test[:,17])) + vec1(abs(X_test[:,17]))*(X_test[:,15]/X_test[:,13])*(X_test[:,16]/X_test[:,13])*( 1 + vec2(fac(abs(X_test[:,17])))*(X_test[:,16]/X_test[:,13]))

mcorrval = vec0(abs(X_test[:,17]))   +  vec1(abs(X_test[:,17]))*eop*dop*( 1   +  vec2(abs(X_test[:,17]))*dop)


# In[5]:


def propweights(y_true):
    weight = np.copy(y_true)
    weight[abs(y_true - meany) > 0] = 1.90*abs(y_true - meany)/stdy  #1.25
#    weight[abs(y_true - meany) > stdy] = 1.75*abs((weight[abs(y_true - meany) > stdy]) - meany)/(stdy)
    weight[abs(y_true - meany) < stdy] =  1
    return weight


# In[6]:


from tensorflow.keras import optimizers
print ("creating model=========>")
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(500, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(60, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1))

RMS = tensorflow.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
# Compile model
print ("compilation up next=======>")
model.compile(loss='logcosh',optimizer='adam')
model.summary()
#fitting
print ("fitting now=========>")
history = model.fit(X_train,Y_train , batch_size=5000, epochs=100, validation_split=0.2, verbose=1,sample_weight=propweights(Y_train))


# In[7]:
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(modv+'_loss_distributions.png')
plt.close()


preds = model.predict(X_test[:,0:12])
targets = X_test[:,12]
uncorrected = X_test[:,15]
marinascorr = X_test[:,15]*mcorrval
plt.hist(preds, bins =100, range=(0,200),label='PU regression',alpha=0.6)
plt.hist(targets, bins =100, range=(0,200),label='no PU',alpha=0.6)
plt.hist(uncorrected, bins =100, range=(0,200),label='uncorrected',alpha=0.6)
#plt.hist(marinascorr, bins =100, range=(0,200),label='marinas correction',alpha=0.6)
plt.yscale('log')
plt.title("Energy distribution")
plt.legend(loc='upper right')
plt.savefig(modv+'_energy_distributions.png')
plt.close()

preds = preds.reshape(preds.shape[0])
print (preds.shape)



#plt.hist(preds, bins =100, range=(0,100),label='PU regression',alpha=0.6)
#plt.hist(abs(preds -targets)/targets , bins =100, range=(0,40),label='PU regression',alpha=0.6)
#plt.hist(targets, bins =100, range=(0,100),label='no PU',alpha=0.6)
plt.hist(abs(uncorrected -targets)/targets*100, bins =100, range=(0,100),label='uncorrected',alpha=0.6)
#plt.hist(abs(marinascorr -targets)/targets*100, bins =100, range=(0,100),label='marinas correction',alpha=0.6)
plt.hist(100*abs(preds -targets)/targets, bins =100, range=(0,100),label='PU correction',alpha=0.6)
#plt.yscale('log')
plt.title("error distribution")
plt.legend(loc='upper right')
plt.savefig(modv+'_errors.png')
plt.close()

#plt.scatter(targets, marinascorr,alpha=0.3,label='marinascorr')
plt.scatter(targets, uncorrected,alpha=0.3,label='uncorr')
plt.scatter(targets, preds,alpha=0.3,label='PUcorr')
plt.xlabel('True Values ')
plt.ylabel('Predictions ')
plt.legend(loc='upper right')
lims = [0, 200]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig(modv+'_energyscatt.png')
plt.close()



pmom= X_test[:,13]
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(targets/pmom, bins =100, range=(0,5),label='E/p noPU',alpha=0.6)
#plt.hist(preds/pmom, bins =100, range=(0,5),label='E/p PUcorr',alpha=0.6)
#plt.hist(uncorrected/pmom, bins =100, range=(0,5),label='E/p uncorr',alpha=0.6)
plt.hist(marinascorr/pmom, bins =100, range=(0,5),label='E/p marina corr',alpha=0.6)
#plt.hist(np.exp(preds.reshape((preds.shape[0])))/pmom[0:n_test_events], bins =100, range=(0,5),label='corrEnp/p',alpha=0.3)
#plt.hist(preds.reshape((preds.shape[0]))/pmom[0:n_test_events], bins =100, range=(0,5),label='corrEnp/p',alpha=0.3)
plt.legend(loc='upper right')
plt.yscale('log')
plt.title("E/p distributions") 
plt.savefig(modv+'_eopdist.png')
plt.close()

# In[8]:


import os
############## save model
if not os.path.exists('models'):
    os.makedirs('models')
model.save('models/model'+modv+'.h5')
#new_model_2 = load_model('my_model.h5')

