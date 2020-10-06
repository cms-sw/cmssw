######################################################################################
# Makes pkl, root and text files comparing PU and noPU samples for training regressor and other stuff
# Usage:
# python3 isotrackRootTreeMaker.py -PU root://cmseos.fnal.gov//store/user/sghosh/ISOTRACK/DIPI_2021_PUpart.root -NPU root://cmseos.fnal.gov//store/user/sghosh/ISOTRACK/DIPI_2021_noPU.root -O isotrackRelval 
######################################################################################

import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument("-PU", "--filePU",help="input PU file",default="2021PU.root")
parser.add_argument("-NPU", "--fileNPU",help="input no PU file",default="2021noPU.root")
parser.add_argument("-O", "--opfilename",help="ouput file name",default="isotk_relval")

fName1 = parser.parse_args().filePU
fName2 = parser.parse_args().fileNPU
foutput = parser.parse_args().opfilename

# PU
tree1 = uproot.open(fName1,xrootdsource=dict(chunkbytes=1024**3, limitbytes=1024**3))['HcalIsoTrkAnalyzer/CalibTree']

#no PU
tree2 = uproot.open(fName2,xrootdsource=dict(chunkbytes=1024**3, limitbytes=1024**3))['HcalIsoTrkAnalyzer/CalibTree']

#tree2.keys()
print ("loaded files")
branchespu = ['t_Run','t_Event','t_nVtx','t_ieta','t_iphi','t_p','t_pt','t_gentrackP','t_eMipDR','t_eHcal','t_eHcal10','t_eHcal30','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh','t_selectTk','t_qltyFlag']
branchesnpu = ['t_Event','t_ieta','t_iphi','t_p','t_eHcal','t_eHcal10','t_eHcal30']
#dictn = tree.arrays(branches=branches,entrystart=0, entrystop=300)
dictpu = tree1.arrays(branches=branchespu)
dictnpu = tree2.arrays(branches=branchesnpu)
dfspu = pd.DataFrame.from_dict(dictpu)
dfspu.columns=branchespu
dfsnpu = pd.DataFrame.from_dict(dictnpu)
dfsnpu.columns=branchesnpu
print ("loaded dicts and dfs")
print ("PU sample size:",dfspu.shape[0])
print ("noPU sample size:",dfsnpu.shape[0])
#dfspu
merged = pd.merge(dfspu, dfsnpu , on=['t_Event','t_ieta','t_iphi'])
print ("selected common events before cut:",merged.shape[0])
#print(merged)
keepvars =  ['t_nVtx','t_ieta','t_eHcal10_x','t_eHcal30_x','t_delta_x','t_eHcal10_y','t_eHcal30_y','t_delta_y','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh','t_pt','t_eHcal_x','t_eHcal_y','t_p_x','t_p_y','t_eMipDR']

'''
#########################all ietas
cuts1 = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<20)&(merged['t_eMipDR']<1)&(abs(merged['t_p'] - 50)<10)&(merged['t_eHcal_x']>10)
merged1=merged.loc[cuts1]
merged1 = merged1.reset_index(drop=True)
print ("selected events after cut for all ietas:",merged1.shape[0])
merged1['t_delta']=merged1['t_eHcal30']-merged1['t_eHcal10']
final_df_all = merged1[keepvars]
#final_dfnp = final_df.values
#np.save('isotk_relval_all.npy',final_df_all.values)
#np.save('isotk_relval_all.npy',final_df_all)
final_df_all.to_pickle(foutput+"_all.pkl")
final_df_all.to_csv(foutput+"_all.txt")
#########################split ieta < 16

cuts2 = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<20)&(merged['t_eMipDR']<1)&(abs(merged['t_ieta'])<16)&(abs(merged['t_p'] - 50)<10)&(merged['t_eHcal_x']>10)
merged2=merged.loc[cuts2]
merged2 = merged2.reset_index(drop=True)
print ("selected events after cut for ieta < 16:",merged2.shape[0])
merged2['t_delta']=merged2['t_eHcal30']-merged2['t_eHcal10']
final_df_low = merged2[keepvars]
#final_dfnp = final_df.values
#np.save('isotk_relval_lo.npy',final_df_low.values)
#np.save('isotk_relval_lo.npy',final_df_low)
final_df_low.to_pickle(foutput+"_lo.pkl")
final_df_low.to_csv(foutput+"_lo.txt")
'''

#########################split ieta > 24

cuts3 = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<20)&(merged['t_eMipDR']<1)&(abs(merged['t_p_x'] - 50)<10)&(merged['t_eHcal_x']>10)
merged3=merged.loc[cuts3]
merged3 = merged3.reset_index(drop=True)
print ("selected events after cut for ieta > 24:",merged3.shape[0])
merged3['t_delta_x']=merged3['t_eHcal30_x']-merged3['t_eHcal10_x']
merged3['t_delta_y']=merged3['t_eHcal30_y']-merged3['t_eHcal10_y']

final_df_hi = merged3[keepvars]
final_df_hi.to_pickle(foutput+"_hi.pkl")
final_df_hi.to_csv(foutput+"_hi.txt")

'''
threedee = plt.figure().gca(projection='3d')
threedee.scatter(final_df_hi['t_eHcal_x'], final_df_hi['t_eHcal_y'], final_df_hi['t_delta'])
threedee.set_xlabel('Corrected Energy')
threedee.set_ylabel('Uncorrected Energy')
threedee.set_zlabel('delta')
fig = threedee.get_figure()
fig.show()
fig.savefig('debu.png')

print(type(merged3['t_p']))
print(merged3['t_p'])
print(merged3['t_p'].to_numpy())

a=merged3['t_p'].to_numpy()
print(type(a))
print(a.ndim)
print(a.shape)
'''

print(merged3['t_ieta'].dtype)

with uproot.recreate(foutput+".root") as f:

    f["tree"] = uproot.newtree({"t_Event": np.int32,
                                "t_p_PU": np.float64,
                                "t_eHcal_PU":np.float64,
                                "t_delta_PU":np.float64,
                                "t_p_NoPU": np.float64,
                                "t_eHcal_noPU":np.float64,
                                "t_delta_NoPU":np.float64,
                                "t_ieta":np.int32})


    f["tree"].extend({"t_Event": merged3['t_Event'],
                      "t_p_PU": merged3['t_p_x'].to_numpy(),
                      "t_eHcal_PU": merged3['t_eHcal_x'].to_numpy(),
                      "t_delta_PU": merged3['t_delta_x'].to_numpy(),
                      "t_p_NoPU": merged3['t_p_y'].to_numpy(),
                      "t_eHcal_noPU": merged3['t_eHcal_y'].to_numpy(),
                      "t_delta_NoPU": merged3['t_delta_y'].to_numpy(),
                      "t_ieta": merged3['t_ieta'].to_numpy()})
