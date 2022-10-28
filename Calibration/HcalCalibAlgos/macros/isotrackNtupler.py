######################################################################################
# Makes pkl and text files comparing PU and noPU samples for training regressor and other stuff
# Usage:
# source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc8-opt/setup.csh
# python3 isotrackNtupler.py -PU root://eoscms.cern.ch//eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/SinglePion_E-50_Eta-0to3_Run3Winter21_112X_PU.root -NPU root://eoscms.cern.ch//eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/SinglePion_E-50_Eta-0to3_Run3Winter21_112X_PU.root -O isotk_relval 
######################################################################################



import uproot
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-PU", "--filePU",help="input PU file",default="root://eoscms.cern.ch//eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/SinglePion_E-50_Eta-0to3_Run3Winter21_112X_PU.root")
parser.add_argument("-NPU", "--fileNPU",help="input no PU file",default="//eoscms.cern.ch//eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/SinglePion_E-50_Eta-0to3_Run3Winter21_112X_PU.root")
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
branchesnpu = ['t_Event','t_ieta','t_iphi','t_eHcal']
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
merged = pd.merge(dfspu, dfsnpu , on=['t_Event','t_ieta','t_iphi'])
print ("selected common events before cut:",merged.shape[0])

#cuts = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<10)&(merged['t_eMipDR_y']<1)
keepvars =  ['t_nVtx','t_ieta','t_eHcal10','t_eHcal30','t_delta','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh','t_pt','t_eHcal_x','t_eHcal_y','t_p','t_eMipDR']



#########################all ietas
cuts1 = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<20)&(merged['t_eMipDR']<1)&(abs(merged['t_p'] - 50)<10)&(merged['t_eHcal_x']>10)
merged1=merged.loc[cuts1]
merged1 = merged1.reset_index(drop=True)
print ("selected events after cut for all ietas:",merged1.shape[0])
merged1['t_delta']=merged1['t_eHcal30']-merged1['t_eHcal10']
final_df_all = merged1[keepvars]
final_df_all.to_pickle(foutput+"_all.pkl")
final_df_all.to_csv(foutput+"_all.txt")
#########################split ieta < 16

cuts2 = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<20)&(merged['t_eMipDR']<1)&(abs(merged['t_ieta'])<16)&(abs(merged['t_p'] - 50)<10)&(merged['t_eHcal_x']>10)
merged2=merged.loc[cuts2]
merged2 = merged2.reset_index(drop=True)
print ("selected events after cut for ieta < 16:",merged2.shape[0])
merged2['t_delta']=merged2['t_eHcal30']-merged2['t_eHcal10']
final_df_low = merged2[keepvars]
final_df_low.to_pickle(foutput+"_lo.pkl")
final_df_low.to_csv(foutput+"_lo.txt")

#########################split ieta > 15

cuts3 = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<20)&(merged['t_eMipDR']<1)&(abs(merged['t_ieta'])>15)&(abs(merged['t_p'] - 50)<10)&(merged['t_eHcal_x']>10)
merged3=merged.loc[cuts3]
merged3 = merged3.reset_index(drop=True)
print ("selected events after cut for ieta > 15:",merged3.shape[0])
merged3['t_delta']=merged3['t_eHcal30']-merged3['t_eHcal10']
final_df_hi = merged3[keepvars]
final_df_hi.to_pickle(foutput+"_hi.pkl")
final_df_hi.to_csv(foutput+"_hi.txt")

