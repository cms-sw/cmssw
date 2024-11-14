# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

## Here we define higher (>50k events) stats data workflows
## not to be run as default. 150k, 250k, 500k or 1M events each

offset_era = 0.1 # less than 10 eras per year
offset_pd = 0.001 # less than 100 pds per year
offset_events = 0.0001 # less than 10 event setups (50k,150k,250k,500k)

## 2024
base_wf = 2024.0
for e_n,era in enumerate(eras_2024):
    for p_n,pd in enumerate(pds_2024):
        for e_key,evs in event_steps_dict.items():
            if "10k" == e_key: # already defined in relval_standard
                continue   
            wf_number = base_wf
            wf_number = wf_number + offset_era * e_n
            wf_number = wf_number + offset_pd * p_n
            wf_number = wf_number + offset_events * evs 
            wf_number = round(wf_number,6)

            step_name = "Run" + pd.replace("ParkingDouble","Park2") + era.split("Run")[1] + "_" + e_key
            y = str(int(base_wf))
            suff = "ZB_" if "ZeroBias" in step_name else ""
            workflows[wf_number] = ['',[step_name,'HLTDR3_' + y,'RECONANORUN3_' + suff + 'reHLT_'+y,'HARVESTRUN3_' + suff + y]]

## 2023
base_wf = 2023.0
for e_n,era in enumerate(eras_2023):
    for p_n,pd in enumerate(pds_2023):
        for e_key,evs in event_steps_dict.items():
            if "10k" == e_key: # already defined in relval_standard
                continue   
            wf_number = base_wf
            wf_number = wf_number + offset_era * e_n
            wf_number = wf_number + offset_pd * p_n
            wf_number = wf_number + offset_events * evs 
            wf_number = round(wf_number,6)

            step_name = "Run" + pd.replace("ParkingDouble","Park2") + era.split("Run")[1] + "_" + e_key
            y = str(int(base_wf)) + "B" if "2023B" in era else str(int(base_wf))
            suff = "ZB_" if "ZeroBias" in step_name else ""
            workflows[wf_number] = ['',[step_name,'HLTDR3_' + y,'RECONANORUN3_' + suff + 'reHLT_'+y,'HARVESTRUN3_' + suff + y]]

## 2022
base_wf = 2022.0
for e_n,era in enumerate(eras_2022_1):
    for p_n,pd in enumerate(pds_2022_1):
        for e_key,evs in event_steps_dict.items():
            if "10k" == e_key: # already defined in relval_standard
                continue   
            wf_number = base_wf
            wf_number = wf_number + offset_era * e_n
            wf_number = wf_number + offset_pd * p_n
            wf_number = wf_number + offset_events * evs
            wf_number = round(wf_number,6)
            step_name = "Run" + pd + era.split("Run")[1] + "_10k"
            y = str(int(base_wf))
            suff = "ZB_" if "ZeroBias" in step_name else ""
            workflows[wf_number] = ['',[step_name,'HLTDR3_' + y,'RECONANORUN3_' + suff + 'reHLT_'+y,'HARVESTRUN3_' + suff + y]]

# PD names changed during 2022
for e_n,era in enumerate(eras_2022_2):
    for p_n,pd in enumerate(pds_2022_2):
        for e_key,evs in event_steps_dict.items():
            if "10k" == e_key: # already defined in relval_standard
                continue   
            wf_number = base_wf
            wf_number = wf_number + offset_era * (e_n + len(eras_2022_1))
            wf_number = wf_number + offset_pd * (p_n + len(pds_2022_1))
            wf_number = wf_number + offset_events * evs 
            wf_number = round(wf_number,6)
            step_name = "Run" + pd + era.split("Run")[1] + "_" + e_key
            y = str(int(base_wf))
            suff = "ZB_" if "ZeroBias" in step_name else ""
            workflows[wf_number] = ['',[step_name,'HLTDR3_' + y,'RECONANORUN3_' + suff + 'reHLT_'+y,'HARVESTRUN3_' + suff + y]] 
