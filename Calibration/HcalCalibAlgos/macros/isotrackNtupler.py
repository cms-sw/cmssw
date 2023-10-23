######################################################################################
# Makes pkl and text files comparing PU and noPU samples for training regressor and other stuff
# Usage:
# source /cvmfs/sft.cern.ch/lcg/views/LCG_97apython3/x86_64-centos7-gcc8-opt/setup.csh
# python3 isotrackNtupler.py -PU root://eoscms.cern.ch//eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/SinglePion_E-50_Eta-0to3_Run3Winter21_112X_PU.root -NPU root://eoscms.cern.ch//eos/cms/store/group/dpg_hcal/comm_hcal/ISOTRACK/SinglePion_E-50_Eta-0to3_Run3Winter21_112X_PU.root -O isotk_relval 
######################################################################################

import uproot3
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-PU", "--filePU", help="input PU file")
parser.add_argument("-NPU", "--fileNPU", help="input no PU file")
parser.add_argument("-O", "--opfilename", help="ouput file name")
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

print ("loaded files")

branchespu = ['t_Run','t_Event','t_nVtx','t_ieta','t_iphi','t_p','t_pt','t_gentrackP','t_eMipDR','t_eHcal','t_eHcal10','t_eHcal30','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh','t_selectTk','t_qltyFlag']

#branchesnpu = ['t_Event','t_ieta','t_iphi','t_eHcal']

branchesnpu =['t_Run','t_Event','t_nVtx','t_ieta','t_iphi','t_p','t_pt','t_gentrackP','t_eMipDR','t_eHcal','t_eHcal10','t_eHcal30','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh','t_selectTk','t_qltyFlag']

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
    branches_skim = ['t_Event','t_ieta','t_iphi','t_eHcal']
    dfsnpu = dfsnpu[branches_skim]

    merged = pd.merge(dfspu, dfsnpu , on=['t_Event','t_ieta','t_iphi'])
    print(merged.keys())
    print ("selected common events before cut:",merged.shape[0])
    
    #cuts = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<10)&(merged['t_eMipDR_y']<1)
    keepvars =  ['t_nVtx','t_ieta','t_eHcal10','t_eHcal30','t_delta','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh','t_pt','t_eHcal_x','t_eHcal_y','t_p','t_eMipDR']



    #########################all ietas
    #cuts1 = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<20)&(merged['t_eMipDR']<1)&(abs(merged['t_p'] - 50)<10)&(merged['t_eHcal_x']>10)

    merged1=merged
    #merged1 = merged1.reset_index(drop=True)

    print ("selected events after cut for all ietas:",merged1.shape[0])
    merged1['t_delta']=merged1['t_eHcal30']-merged1['t_eHcal10']
    final_df_all = merged1[keepvars]
    output_file = foutput+'_'+str(i)+"_"+start+"_"+stop+"_all.parquet"
    final_df_all.to_parquet(output_file)
    final_df_all.to_csv(foutput+"_"+str(i)+"_"+start+"_"+stop+"_all.txt")

    #########################split ieta < 16
    
    cuts2 = abs(merged['t_ieta'])<16
    merged2=merged.loc[cuts2]
    merged2 = merged2.reset_index(drop=True)
    print ("selected events after cut for ieta < 16:",merged2.shape[0])

    merged2['t_delta']=merged2['t_eHcal30']-merged2['t_eHcal10']
    final_df_low = merged2[keepvars]
    final_df_low.to_parquet(foutput+'_'+str(i)+"_"+start+"_"+stop+"_lo.parquet")
    final_df_low.to_csv(foutput+'_'+str(i)+"_"+start+"_"+stop+"_lo.txt")

    #########################split ieta > 15
    
    cuts3 = abs(merged['t_ieta'])>15
    merged3=merged.loc[cuts3]
    merged3 = merged3.reset_index(drop=True)
    print ("selected events after cut for ieta > 15:",merged3.shape[0])

    merged3['t_delta']=merged3['t_eHcal30']-merged3['t_eHcal10']
    final_df_hi = merged3[keepvars]
    final_df_hi.to_parquet(foutput+'_'+str(i)+"_"+start+"_"+stop+"_hi.parquet")
    final_df_hi.to_csv(foutput+'_'+str(i)+"_"+start+"_"+stop+"_hi.txt")
    i+=1
