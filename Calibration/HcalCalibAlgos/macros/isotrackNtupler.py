import uproot
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
tree1 = uproot.open(fName1)['hcalIsoTrkAnalyzer/CalibTree']

#no PU
tree2 = uproot.open(fName2)['hcalIsoTrkAnalyzer/CalibTree']

print ("loaded files")

branchespu = ['t_Run','t_Event','t_nVtx','t_ieta','t_iphi','t_p','t_pt','t_gentrackP','t_eMipDR','t_eHcal','t_eHcal10','t_eHcal30','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh','t_selectTk','t_qltyFlag', 't_EventWeight', 't_DetIds', 't_HitEnergies']

branchesnpu = ['t_Event','t_ieta','t_iphi','t_eHcal', 't_DetIds', 't_HitEnergies']

dictpu = tree1.arrays(branchespu, entry_start=int(start), entry_stop=int(stop))
#dictpu = {key.decode(): val for key, val in dictpu.items()}

npu_entries = tree2.num_entries

scale = 500000
npu_start = 0
i = 0

for index in range(0,npu_entries, scale):
    npu_stop = index+scale
    if (npu_stop > npu_entries):
        npu_stop = npu_entries
    dictnpu = tree2.arrays(branchesnpu, entry_start=npu_start, entry_stop=npu_stop)
    #dictnpu = {key.decode(): val for key, val in dictnpu.items()}
    
    npu_start = npu_stop

    dfspu = pd.DataFrame({col: dictpu[col] for col in branchespu[:-2]}) #.from_dict(dictpu[branchespu[:-2]])
    dfspu.columns=branchespu[:-2]
    dfsnpu = pd.DataFrame({col: dictnpu[col] for col in branchesnpu[:-2]})
    dfsnpu.columns=branchesnpu[:-2]
    
    print("loaded % of nopile file is =",(npu_stop/npu_entries)*100)
    print ("PU sample size:",dfspu.shape[0])
    print ("noPU sample size:",dfsnpu.shape[0])

    dfspu['idx_spu'] = dfspu.index
    dfsnpu['idx_npu'] = dfsnpu.index
    
    merged = pd.merge(dfspu, dfsnpu , on=['t_Event','t_ieta','t_iphi'])
    print ("selected common events before cut:",merged.shape[0])
    
    keepvars =  ['t_nVtx','t_ieta','t_eHcal10','t_eHcal30','t_delta','t_hmaxNearP','t_emaxNearP','t_hAnnular','t_eAnnular','t_rhoh','t_pt','t_eHcal_x','t_eHcal_y','t_p','t_eMipDR', 't_EventWeight']


    cuts1 = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<20)&(merged['t_eMipDR']<1)&(abs(merged['t_p'] - 50)<10)&(merged['t_eHcal_x']>10)

    merged1=merged.loc[cuts1]
    merged1 = merged1.reset_index(drop=True)

    print ("selected events after cut for all ietas:",merged1.shape[0])
    if(merged1.shape[0] == 0):
        i +=1
        continue
    merged1['t_delta']=merged1['t_eHcal30']-merged1['t_eHcal10']
    final_df_all = merged1[keepvars]
    output_file = foutput+'_'+str(i)+"_"+start+"_"+stop+"_all.root"
    root_data = {}
    for column in final_df_all.columns:
        root_data[column] = final_df_all[column].values

    HitEnergies_x = dictpu[merged1["idx_spu"].values]["t_HitEnergies"]
    HitEnergies_y = dictnpu[(merged1["idx_npu"].values)]["t_HitEnergies"]
    DetIds_x = dictpu[merged1["idx_spu"].values]["t_DetIds"]
    DetIds_y = dictnpu[merged1["idx_npu"].values]["t_DetIds"]
    root_data["t_hitEnergies_x"] = HitEnergies_x
    root_data["t_hitEnergies_y"] = HitEnergies_y
    root_data["t_DetIds_x"] = DetIds_x
    root_data["t_DetIds_y"] = DetIds_y
    with uproot.recreate(output_file) as f:
        f["CalibTree"] = root_data  
    i+=1


    cuts1 = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<20)&(merged['t_eMipDR']<1)&(abs(merged['t_p'] - 50)<10)&(merged['t_eHcal_x']>10) &(abs(merged['t_ieta']) <16)

    merged1=merged.loc[cuts1]
    merged1 = merged1.reset_index(drop=True)

    print ("selected events after cut for all ietas:",merged1.shape[0])
    if(merged1.shape[0] == 0):
        i +=1
        continue
    merged1['t_delta']=merged1['t_eHcal30']-merged1['t_eHcal10']
    final_df_all = merged1[keepvars]
    output_file = foutput+'_'+str(i)+"_"+start+"_"+stop+"_barrel.root"
    root_data = {}
    for column in final_df_all.columns:
        root_data[column] = final_df_all[column].values

    HitEnergies_x = dictpu[merged1["idx_spu"].values]["t_HitEnergies"]
    HitEnergies_y = dictnpu[(merged1["idx_npu"].values)]["t_HitEnergies"]
    DetIds_x = dictpu[merged1["idx_spu"].values]["t_DetIds"]
    DetIds_y = dictnpu[merged1["idx_npu"].values]["t_DetIds"]
    root_data["t_hitEnergies_x"] = HitEnergies_x
    root_data["t_hitEnergies_y"] = HitEnergies_y
    root_data["t_DetIds_x"] = DetIds_x
    root_data["t_DetIds_y"] = DetIds_y
    with uproot.recreate(output_file) as f:
        f["CalibTree"] = root_data  
    i+=1


    cuts1 = (merged['t_selectTk'])&(merged['t_qltyFlag'])&(merged['t_hmaxNearP']<20)&(merged['t_eMipDR']<1)&(abs(merged['t_p'] - 50)<10)&(merged['t_eHcal_x']>10) &(abs(merged['t_ieta']) >15)

    merged1=merged.loc[cuts1]
    merged1 = merged1.reset_index(drop=True)

    print ("selected events after cut for all ietas:",merged1.shape[0])
    if(merged1.shape[0] == 0):
        i +=1
        continue
    merged1['t_delta']=merged1['t_eHcal30']-merged1['t_eHcal10']
    final_df_all = merged1[keepvars]
    output_file = foutput+'_'+str(i)+"_"+start+"_"+stop+"_ee.root"
    root_data = {}
    for column in final_df_all.columns:
        root_data[column] = final_df_all[column].values

    HitEnergies_x = dictpu[merged1["idx_spu"].values]["t_HitEnergies"]
    HitEnergies_y = dictnpu[(merged1["idx_npu"].values)]["t_HitEnergies"]
    DetIds_x = dictpu[merged1["idx_spu"].values]["t_DetIds"]
    DetIds_y = dictnpu[merged1["idx_npu"].values]["t_DetIds"]
    root_data["t_hitEnergies_x"] = HitEnergies_x
    root_data["t_hitEnergies_y"] = HitEnergies_y
    root_data["t_DetIds_x"] = DetIds_x
    root_data["t_DetIds_y"] = DetIds_y
    with uproot.recreate(output_file) as f:
        f["CalibTree"] = root_data  
    i+=1
