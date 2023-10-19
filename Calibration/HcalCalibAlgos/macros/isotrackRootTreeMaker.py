######################################################################################
# Makes pkl, root and text files comparing PU and noPU samples for training regressor and other stuff
# Usage:
# source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc8-opt/setup.csh
# python3 isotrackRootTreeMaker.py -PU root://eoscms.cern.ch//eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/SinglePion_E-50_Eta-0to3_Run3Winter21_112X_PU.root -NPU root://eoscms.cern.ch//eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/SinglePion_E-50_Eta-0to3_Run3Winter21_112X_PU.root -O isotrackRelval 
######################################################################################

import uproot3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()

parser.add_argument("-PU", "--filePU",help="input PU file")
parser.add_argument("-NPU", "--fileNPU",help="input no PU file")
parser.add_argument("-O", "--opfilename",help="ouput file name")
parser.add_argument("-s", "--start", help="start entry for input PU file")
parser.add_argument("-e", "--end", help="end entry for input PU file")


fName1 = parser.parse_args().filePU
fName2 = parser.parse_args().fileNPU
foutput = parser.parse_args().opfilename
start = parser.parse_args().start
stop = parser.parse_args().end

# PU
tree1 = uproot3.open(fName1)['hcalIsoTrkAnalyzer/CalibTree']

#no PU
tree2 = uproot3.open(fName2)['hcalIsoTrkAnalyzer/CalibTree']

#tree2.keys()
print ("loaded files")

branchespu = ['t_Run','t_Event','t_nVtx','t_ieta','t_iphi','t_p','t_pt','t_gentrackP','t_eMipDR','t_eHcal','t_eHcal10','t_eHcal30','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh','t_selectTk','t_qltyFlag']

branchesnpu = ['t_Run','t_Event','t_nVtx','t_ieta','t_iphi','t_p','t_pt','t_gentrackP','t_eMipDR','t_eHcal','t_eHcal10','t_eHcal30','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh','t_selectTk','t_qltyFlag']

dictpu = tree1.arrays(branchespu, entrystart=int(start), entrystop=int(stop))

npu_entries = tree2.numentries

scale = 5000000
npu_start = 0
i = 0

for index in range(0,npu_entries, scale):
    npu_stop = index+scale
    if (npu_stop > npu_entries):
        npu_stop = npu_entries
    dictnpu = tree2.arrays(branchesnpu, entrystart=npu_start, entrystop=npu_stop)
    npu_start = npu_stop
    dfspu = pd.DataFrame.from_dict(dictpu)
    dfspu.columns=branchespu
    dfsnpu = pd.DataFrame.from_dict(dictnpu)
    dfsnpu.columns=branchesnpu
    print("loaded % of nopile file is =",(npu_stop/npu_entries)*100)
    print ("PU sample size:",dfspu.shape[0])
    print ("noPU sample size:",dfsnpu.shape[0])
    
    cuts_pu = (dfspu['t_selectTk'])&(dfspu['t_qltyFlag'])&(dfspu['t_hmaxNearP']<20)&(dfspu['t_eMipDR']<1)&(abs(dfspu['t_p'] - 50)<10)&(dfspu['t_eHcal']>10)

    cuts_npu = (dfsnpu['t_selectTk'])&(dfsnpu['t_qltyFlag'])&(dfsnpu['t_hmaxNearP']<20)&(dfsnpu['t_eMipDR']<1)&(abs(dfsnpu['t_p'] - 50)<10)&(dfsnpu['t_eHcal']>10)

    dfspu = dfspu.loc[cuts_pu]
    dfspu = dfspu.reset_index(drop=True)

    dfsnpu = dfsnpu.loc[cuts_npu]
    dfsnpu = dfsnpu.reset_index(drop=True)
    branches_skim = ['t_Event','t_ieta','t_iphi','t_p','t_eHcal','t_eHcal10','t_eHcal30']
    dfsnpu = dfsnpu[branches_skim]

    merged = pd.merge(dfspu, dfsnpu , on=['t_Event','t_ieta','t_iphi'])
    print ("selected common events before cut:",merged.shape[0])
    #print(merged)
    keepvars =  ['t_nVtx','t_ieta','t_eHcal10_x','t_eHcal30_x','t_delta_x','t_eHcal10_y','t_eHcal30_y','t_delta_y','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh','t_pt','t_eHcal_x','t_eHcal_y','t_p_x','t_p_y','t_eMipDR']

    merged3 = merged
    #merged3 = merged3.reset_index(drop=True)
    print ("selected events after cut:",merged3.shape[0])
    merged3['t_delta_x']=merged3['t_eHcal30_x']-merged3['t_eHcal10_x']
    merged3['t_delta_y']=merged3['t_eHcal30_y']-merged3['t_eHcal10_y']

    final_df_hi = merged3[keepvars]
    final_df_hi.to_parquet(foutput+'_'+str(i)+"_"+start+"_"+stop+".parquet")
    final_df_hi.to_csv(foutput+'_'+str(i)+"_"+start+"_"+stop+".txt")
    
    print(merged3['t_ieta'].dtype)

    with uproot3.recreate(foutput+'_'+str(i)+"_"+start+"_"+stop+".root") as f:

        f["tree"] = uproot3.newtree({"t_Event": np.int32,
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
    i += 1
