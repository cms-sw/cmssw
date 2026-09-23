#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time


# In[2]:


from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# In[3]:


#!pip install hist --only-binary=:all:


# In[4]:


import uproot
import awkward as ak
import matplotlib.pyplot as plt


# In[5]:


import mplhep, hist
plt.style.use([mplhep.style.CMS])
import mplhep as hep


# In[6]:


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
import ROOT
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.models import save_model
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler


# In[7]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print("device:", device)


# In[8]:


branches = [
    "t_nVtx",
    "t_ieta",
    "t_eHcal10",
    "t_eHcal30",
    "t_delta",
    "t_hmaxNearP",
    "t_emaxNearP",
    "t_hAnnular",
    "t_eAnnular",
    "t_rhoh",
    "t_pt",
    "t_eHcal_x",
    "t_eHcal_y",
    "t_p",
    "t_eMipDR",
]


# In[9]:


get_ipython().system('ls /eos/cms/store/group/dpg_hcal/comm_hcal/suman/13p6TeV/Run3Winter2025_MOY/')


# In[10]:


import onnxruntime
_default_session_options = onnxruntime.capi._pybind_state.get_default_session_options()


# In[11]:


model = tf.keras.models.load_model("/eos/user/d/dasgupsu/alcareco_treemaker/collision_13.6TeV/CMSSW_13_1_0_pre3/src/Calibration/HcalCalibAlgos/macros/checkpointsVF/model_best.keras")
model.export("exported_savedmodel")


# In[12]:


get_ipython().system('python -m tf2onnx.convert      --saved-model exported_savedmodel      --output /eos/user/d/dasgupsu/alcareco_treemaker/collision_13.6TeV/CMSSW_13_1_0_pre3/src/Calibration/HcalCalibAlgos/macros/checkpointsVF/model_best.onnx      --opset 13')


# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


events = uproot.open("/eos/cms/store/group/dpg_hcal/comm_hcal/suman/13p6TeV/Run3Winter2025_MOY/IsoTrkNtupler_Run3Winter2025_MOY_ALL_CLEAN.root:CalibTree")
tree = events.arrays(branches, library= "pd")


# In[12]:


df = tree
df = df[abs(df["t_ieta"].values) > 15]
df = df[((df["t_nVtx"].values) > 0.5) & ((df["t_nVtx"].values) < 109.5)]


# In[13]:


df


# In[14]:


#np.count_nonzero(abs(df["t_ieta"].values) > 16 )


# In[15]:


#df = pd.read_parquet(fName)
print ("vars in file:",df.keys())
print("events in df original:",df.shape[0])


# In[16]:


print("events in df after energy cut:",df.shape[0])
df['t_eHcal_xun'] = df['t_eHcal_x']
df['t_delta_un'] = df['t_delta']
df['t_ieta_un'] = df['t_ieta']
df['t_eHcal_yun'] = df['t_eHcal_y']


# In[17]:


data_Comm = uproot.open("/eos/user/d/dasgupsu/alcareco_treemaker/collision_13.6TeV/CMSSW_15_1_0_pre5/src/Calibration/HcalCalibAlgos/macros/Results/EGamma_UnCorr_EraG_Part_nVtx.root")


# In[18]:


nVtx_data_hist = data_Comm["h_nVtx"]

# Extract data
nVtx_data = nVtx_data_hist.values()[1:110]
nVtx_edges  = nVtx_data_hist.axes[0].edges()[1:111]


# In[19]:


bin_width = np.diff(nVtx_edges)

count_data = nVtx_data / np.sum(nVtx_data * bin_width)


# In[20]:


#nVtx_edges, count_data


# In[21]:


nVtx_mc = df["t_nVtx"].to_numpy()


# In[22]:


count_mc, bins_mc = np.histogram(
    nVtx_mc,
    bins=nVtx_edges,
    density=True
)


# In[23]:


#count_mc, bins_mc


# In[24]:


weight = count_data/count_mc


# In[25]:


weighted_pileup_mc = weight[(nVtx_mc-1)]


# In[26]:


# Unweighted MC
count_mc_plot, _ = np.histogram(
    nVtx_mc,
    bins=nVtx_edges,
    density=True
)

# Weighted MC
count_mc_weighted, _ = np.histogram(
    nVtx_mc,
    bins=nVtx_edges,
    weights=weighted_pileup_mc,
    density=True
)

bin_centers = 0.5 * (nVtx_edges[1:] + nVtx_edges[:-1])

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

# DATA
plt.errorbar(
    bin_centers,
    count_data,
    yerr=np.sqrt(nVtx_data) / np.sum(nVtx_data * bin_width),
    fmt="o",
    label="Data",
    color="black"
)

# MC (before reweighting)
plt.step(
    bin_centers,
    count_mc_plot,
    where="mid",
    linewidth=2,
    label="MC (original)",
    color="red"
)

# MC (after reweighting)
plt.step(
    bin_centers,
    count_mc_weighted,
    where="mid",
    linewidth=2,
    label="MC (PU reweighted)",
    color="blue"
)

plt.xlabel("N$_{vtx}$")
plt.ylabel("Normalized events")
plt.title("Pileup distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[27]:


df["weights"] = weighted_pileup_mc


# In[28]:


log_vars = ['t_delta', 't_rhoh', 't_eHcal10', 't_eHcal30', 't_hAnnular', 't_eAnnular']


# In[29]:


for v in log_vars:
    df[v] = np.log1p(df[v])


# In[30]:


mask = (abs(df["t_rhoh"].values) < 5) & (abs(df["t_eHcal30"].values) < 10) &  ((df["t_ieta"].values) > -35) & ((df["t_hAnnular"].values) >= 0) & ((df["t_hAnnular"].values) < 50) & ((df["t_eAnnular"].values) >= 0) & ((df["t_eAnnular"].values) < 150) &  ((df["t_ieta"].values) < 35)
df = df[mask]
df.dropna()


# In[31]:


np.count_nonzero(df["weights"].values > 0)


# In[32]:


weights = np.ones_like(df["t_nVtx"].values)
weights = weights/np.sum(weights)
weights = weights*df["weights"].values
weights = weights/np.sum(weights)
weights = weights.reshape(weights.shape[0])


# In[33]:


np.count_nonzero(df["weights"].values > 0)


# In[34]:


#input_var = ['t_delta', 't_hAnnular', 't_eAnnular','t_pt','t_nVtx','t_ieta','t_eHcal10', 't_eHcal30','t_rhoh', 't_eHcal_x']
input_var = ['t_delta', 't_nVtx','t_eHcal10', 't_eHcal30','t_rhoh', 't_eHcal_x']
input_data = df[input_var].values
target_data = df["t_eHcal_y"].values#/df["t_eHcal_x"].values
add_data = df[["t_p","t_ieta_un","t_eHcal_xun","t_eHcal_yun"]].values


# In[35]:


testindx = int(0.7 * input_data.shape[0])

X_train_raw = input_data[:testindx]
X_test_raw  = input_data[testindx:]

Y_train_raw = target_data[:testindx]
Y_test_raw  = target_data[testindx:]

w_train = weights[:testindx]
w_test  = weights[testindx:]

add_train = add_data[:testindx,:]
add_test  = add_data[testindx:,:]

input_mean = np.average(X_train_raw, axis=0, weights=w_train)
input_var  = np.average((X_train_raw - input_mean)**2, axis=0, weights=w_train)
input_std  = np.sqrt(input_var)

target_mean = np.average(Y_train_raw, axis=0, weights=w_train)
target_var  = np.average((Y_train_raw - target_mean)**2, axis=0, weights=w_train)
target_std  = np.sqrt(target_var)

X_train = (X_train_raw - input_mean) / input_std
X_test  = (X_test_raw  - input_mean) / input_std

Y_train = ((Y_train_raw - target_mean) / target_std).reshape(-1,1)
Y_test  = ((Y_test_raw  - target_mean) / target_std).reshape(-1,1)


# In[36]:


target_mean


# In[37]:


target_std


# In[238]:


X_train[:,-1]


# In[239]:


Y_train[:,0]


# In[240]:


fig, ax = plt.subplots(figsize=(10, 6))

# Plot as step histograms
ax.hist(X_train[:,-1], bins=20, histtype='step', linewidth=2, 
        label='X_train[-1]', color='blue')
ax.hist(Y_train[:,0], bins=20, histtype='step', linewidth=2, 
        label='Y_train[:,0]', color='red')

ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.set_title('Step Histogram Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[241]:


weight_matrix_train = np.concatenate((Y_train, w_train.reshape(-1,1)), axis = 1)


# In[242]:


meany = np.mean(Y_train)
print ("mean y:",meany)
stdy = np.std(Y_train)
print ("std y:", stdy)


# In[243]:


weight_matrix_train


# In[244]:


def propweights(weight_matrix):
    y_true = np.copy(weight_matrix[:,0])
    weight1 =  np.copy(weight_matrix[:,0])
    weight2 =  np.copy(weight_matrix[:,1])
    #weight = np.copy(y_true)
    weight1 = weight1.reshape(-1)
    weight2 = weight2.reshape(-1)
    dev = abs(y_true - meany)
    dev = dev.reshape(-1)
    #weight[dev > 0] = 1.90*dev/stdy  #1.25
    weight1[dev > stdy] = 1.05*abs((weight1[dev > stdy]) - meany)/(stdy)
    weight1[dev < stdy] =  1
    #print ("wieght : ", weight1)
    #print ("wieght : ", weight2)
    return weight2*1e7


# In[ ]:





# In[245]:


from keras import optimizers
print ("creating model=========>")
model = Sequential()
model.add(Dense(64, input_shape=(6,), activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1))


# In[246]:


from keras.losses import LogCosh

model.compile(
    loss=LogCosh(),
    optimizer='adam',
    metrics=['mae'],
    weighted_metrics=['mae']
)
model.summary()


# In[247]:


get_ipython().system('pwd')


# In[248]:


from tensorflow.keras.callbacks import ModelCheckpoint
import os

os.makedirs("checkpointsVF", exist_ok=True)

checkpoint_cb = ModelCheckpoint(
    filepath="checkpointsVF/model_epoch_{epoch:03d}.keras",
    monitor="val_loss",
    save_best_only=False,   # save EVERY epoch
    save_weights_only=False,
    verbose=1
)

best_cb = ModelCheckpoint(
    filepath="checkpointsVF/model_best.keras",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)


# In[1]:


get_ipython().system('pwd')


# In[250]:


split = int(0.8 * X_train.shape[0])

X_tr, X_val = X_train[:split], X_train[split:]
Y_tr, Y_val = Y_train[:split], Y_train[split:]

w_tr = propweights(weight_matrix_train[:split])
w_val = propweights(weight_matrix_train[split:])

history = model.fit(
    X_tr[:], Y_tr,
    validation_data=(X_val[:], Y_val, w_val),
    sample_weight=w_tr,
    batch_size=5000,
    epochs=500,
    shuffle=True,
    callbacks=[checkpoint_cb, best_cb]
)


# In[201]:


#print ("fitting now=========>")
#history = model.fit(X_train,Y_train , batch_size=5000, epochs=250, validation_split=0.2, sample_weight=pd.Series(propweights(weight_matrix_train)).to_frame('weight'))


# In[202]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.savefig(modv+'_loss_distributions_lowWinter.png')
plt.show() 
#plt.close()


# In[203]:


preds = model.predict(X_test[:,:10])
targets = Y_test[:,:]


# In[204]:


inputs = X_test[:,-1]


# In[205]:


inputs_en = (inputs*input_std[-1]) + input_mean[-1]
target_en = ((targets*target_std) + target_mean).reshape(-1)#*inputs_en
preds_en = ((preds*target_std) + target_mean).reshape(-1)#*inputs_en


# In[206]:


input_std


# In[207]:


mask = ((abs(add_test[:,1]) == 23) & (add_test[:,0] != 0) & (add_test[:,0] < 100) & (add_test[:,0] > 10))


# In[208]:


#w_test


# In[209]:


target_arr_trans = target_en[mask]/add_test[:,0][mask]
input_arr_trans = inputs_en[mask]/add_test[:,0][mask]
predict_arr_trans = preds_en[mask]/add_test[:,0][mask]
w_arr = w_test[mask]


# In[211]:


import numpy as np

# 2. Define Helper for Weighted Stats
def get_weighted_stats(data, weights):
    mean = np.average(data, weights=weights)
    variance = np.average((data - mean)**2, weights=weights)
    return mean, np.sqrt(variance)

# 3. Calculate Stats
in_mu, in_sigma = get_weighted_stats(input_arr_trans, w_arr)
tar_mu, tar_sigma = get_weighted_stats(target_arr_trans, w_arr)
pred_mu, pred_sigma = get_weighted_stats(predict_arr_trans, w_arr)

# 4. Fill Histograms (Same as before)
input_hist = hist.new.Reg(50, 0, 5, underflow=True, overflow=True).Weight()
target_hist = hist.new.Reg(50, 0, 5, underflow=True, overflow=True).Weight()
predict_hist = hist.new.Reg(50, 0, 5, underflow=True, overflow=True).Weight()

input_hist.fill(input_arr_trans, weight=w_arr)
target_hist.fill(target_arr_trans, weight=w_arr)
predict_hist.fill(predict_arr_trans, weight=w_arr)

# 5. Plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import mplhep as hep

fig, ax = plt.subplots(1, 1, gridspec_kw={'wspace': 0.12, 'hspace': 0.12}, sharex=True)


hep.histplot(
    target_hist,
    label=rf'noPU ($\mu={tar_mu:.3f}, \sigma={tar_sigma:.3f}$)',
    color="black",
    linewidth=1,
    ax=ax
)

hep.histplot(
    input_hist,
    label=rf'PU (uncorr) ($\mu={in_mu:.3f}, \sigma={in_sigma:.3f}$)',
    color="red",
    linewidth=1,
    ax=ax
)

hep.histplot(
    predict_hist,
    label=rf'PU (corr) ($\mu={pred_mu:.3f}, \sigma={pred_sigma:.3f}$)',
    color="blue",
    alpha=1.0,
    ax=ax
)
ax.set_title("Run3Winter2025_MOY_PUCorrection: ieta = 23", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel(r'$E_{\mathrm{HCAL}}/(p_{\mathrm{Track}}-E_{\mathrm{ECAL}})$')
ax.set_ylabel('Normalized Events')
ax.legend(loc="upper right", fontsize=12)
#ax.set_yscale('log')
plt.tight_layout()
fig.savefig("Run3Winter2025_MOY_PUCorrection_EB.png")
plt.show()


# In[105]:


plt.hist(inputs_en, bins =100, range=(0,150),label='PU regression',alpha=0.6)
plt.hist(target_en, bins =100, range=(0,150),label='no PU',alpha=0.6)
plt.hist(preds_en, bins =100, range=(0,150),label='no PU',alpha=0.6)
#plt.hist(uncorrected, bins =100, range=(0,200),label='uncorrected',alpha=0.6)
#plt.hist(marinascorr, bins =100, range=(0,200),label='marinas correction',alpha=0.6)
#plt.yscale('log')
plt.title("Energy distribution")
plt.legend(loc='upper right')
#plt.savefig(modv+'_energy_distributions_lowWinter.png')
plt.show() 


# In[ ]:





# In[ ]:




