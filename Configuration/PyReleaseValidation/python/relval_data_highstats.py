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
base_wf_number_2024 = 2024.0
for e_n,era in enumerate(eras_2024):
    for p_n,pd in enumerate(pds_2024):
        for e_key,evs in event_steps_dict.items():
            if "10k" == e_key: # already defined in relval_standard
                continue   
            wf_number = base_wf_number_2024
            wf_number = wf_number + offset_era * e_n
            wf_number = wf_number + offset_pd * p_n
            wf_number = wf_number + offset_events * evs 
            wf_number = round(wf_number,6)
            step_name = "Run" + pd + era.split("Run")[1] + "_" + e_key
            workflows[wf_number] = ['',[step_name,'HLTDR3_2024','AODNANORUN3_reHLT_2024','HARVESTRUN3_2024']]

## 2023
base_wf_number_2023 = 2023.0
for e_n,era in enumerate(eras_2023):
    for p_n,pd in enumerate(pds_2023):
        for e_key,evs in event_steps_dict.items():
            if "10k" == e_key: # already defined in relval_standard
                continue   
            wf_number = base_wf_number_2023
            wf_number = wf_number + offset_era * e_n
            wf_number = wf_number + offset_pd * p_n
            wf_number = wf_number + offset_events * evs 
            wf_number = round(wf_number,6)
            step_name = "Run" + pd + era.split("Run")[1] + "_" + e_key
            workflows[wf_number] = ['',[step_name,'HLTDR3_2023','AODNANORUN3_reHLT_2023','HARVESTRUN3_2023']]


## 2022
base_wf_number_2022 = 2022.0
for e_n,era in enumerate(eras_2022_1):
    for p_n,pd in enumerate(pds_2022_1):
        for e_key,evs in event_steps_dict.items():
            if "10k" == e_key: # already defined in relval_standard
                continue   
            wf_number = base_wf_number_2022
            wf_number = wf_number + offset_era * e_n
            wf_number = wf_number + offset_pd * p_n
            wf_number = wf_number + offset_events * evs 
            wf_number = round(wf_number,6)
            step_name = "Run" + pd + era.split("Run")[1] + "_" + e_key
            workflows[wf_number] = ['',[step_name,'HLTDR3_2022','AODNANORUN3_reHLT_2022','HARVESTRUN3_2022']]

for e_n,era in enumerate(eras_2022_2):
    for p_n,pd in enumerate(pds_2022_2):
        for e_key,evs in event_steps_dict.items():
            if "10k" == e_key: # already defined in relval_standard
                continue   
            wf_number = base_wf_number_2022
            wf_number = wf_number + offset_era * (e_n + len(eras_2022_1))
            wf_number = wf_number + offset_pd * (p_n + len(pds_2022_1))
            wf_number = wf_number + offset_events * evs 
            wf_number = round(wf_number,6)
            step_name = "Run" + pd + era.split("Run")[1] + "_" + e_key
            workflows[wf_number] = ['',[step_name,'HLTDR3_2022','AODNANORUN3_reHLT_2022','HARVESTRUN3_2022']]



