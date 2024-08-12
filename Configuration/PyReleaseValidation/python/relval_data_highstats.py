# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

## Here we define higher (>50k events) stats data workflows
## not to be run as default. 150k, 250k, 500k or 1M events each

## 2024
base_wf_number_2024 = 2024.0
offset_era = 0.1 # less than 10 eras
offset_pd = 0.001 # less than 100 pds
offset_events = 0.0001 # less than 10 event setups (50k,150k,250k,500k)

for e_n,era in enumerate(eras_2024):
    for p_n,pd in enumerate(pds_2024):
        for e_key,evs in event_steps_dict.items():
            if "50k" == e_key: # already defined in relval_standard
                continue   
            wf_number = base_wf_number_2024
            wf_number = wf_number + offset_era * e_n
            wf_number = wf_number + offset_pd * p_n
            wf_number = wf_number + offset_events * evs 
            wf_number = round(wf_number,6)
            step_name = "Run" + pd + era.split("Run")[1] + "_" + e_key
            workflows[wf_number] = ['',[step_name,'HLTDR3_2024','AODNANORUN3_reHLT_2024','HARVESTRUN3_2024']]



