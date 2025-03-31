# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

## Here we define fixed high stats data workflows
## not to be run as default. 10k, 50k, 150k, 250k, 500k or 1M events each

offset_era = 0.1 # less than 10 eras per year (hopefully!)
offset_pd = 0.001 # less than 100 pds per year
offset_events = 0.0001 # less than 10 event setups (10k,50k,150k,250k,500k,1M)

## 2024
base_wf = 2024.0
for e_n,era in enumerate(eras_2024):
    for p_n,pd in enumerate(pds_2024):
        for e_key,evs in event_steps_dict.items(): 
            wf_number = base_wf
            wf_number = wf_number + offset_era * e_n
            wf_number = wf_number + offset_pd * p_n
            wf_number = wf_number + offset_events * evs 
            wf_number = round(wf_number,6)

            ## Here we use JetMET1 PD to run the TeVJet skims
            skim = 'TeVJet' if pd == 'JetMET1' else ''

            ## ZeroBias have their own HARVESTING
            suff = 'ZB_' if 'ZeroBias' in pd else ''

            # Running C,D,E with the offline GT.
            # Could be removed once 2025 wfs are in and we'll test the online GT with them
            recosetup = 'RECONANORUN3_' + suff + 'reHLT_2024' 
            recosetup = recosetup if era[-1] > 'E' else recosetup + '_Offline'
            
            y = str(int(base_wf))
            step_name = 'Run' + pd.replace('ParkingDouble','Park2') + era.split('Run')[1] + skim + '_' + e_key
            workflows[wf_number] = ['',[step_name,'HLTDR3_' + y,'RECONANORUN3_' + suff + 'reHLT_'+y,'HARVESTRUN3_' + suff + y]]

## 2023
base_wf = 2023.0
for e_n,era in enumerate(eras_2023):
    for p_n,pd in enumerate(pds_2023):
        for e_key,evs in event_steps_dict.items():  
            wf_number = base_wf
            wf_number = wf_number + offset_era * e_n
            wf_number = wf_number + offset_pd * p_n
            wf_number = wf_number + offset_events * evs 
            wf_number = round(wf_number,6)
            step_name = 'Run' + pd.replace('ParkingDouble','Park2') + era.split('Run')[1] + '_' + e_key
            y = str(int(base_wf)) + 'B' if '2023B' in era else str(int(base_wf))
            suff = 'ZB_' if 'ZeroBias' in step_name else ''
            workflows[wf_number] = ['',[step_name,'HLTDR3_' + y,'RECONANORUN3_' + suff + 'reHLT_'+y,'HARVESTRUN3_' + suff + y]]

## 2022
base_wf = 2022.0
for e_n,era in enumerate(eras_2022_1):
    for p_n,pd in enumerate(pds_2022_1):
        for e_key,evs in event_steps_dict.items(): 
            wf_number = base_wf
            wf_number = wf_number + offset_era * e_n
            wf_number = wf_number + offset_pd * p_n
            wf_number = wf_number + offset_events * evs
            wf_number = round(wf_number,6)
            step_name = 'Run' + pd + era.split('Run')[1] + '_' + e_key
            y = str(int(base_wf))
            suff = 'ZB_' if 'ZeroBias' in step_name else ''
            workflows[wf_number] = ['',[step_name,'HLTDR3_' + y,'RECONANORUN3_' + suff + 'reHLT_'+y,'HARVESTRUN3_' + suff + y]]

# PD names changed during 2022
for e_n,era in enumerate(eras_2022_2):
    for p_n,pd in enumerate(pds_2022_2):
        for e_key,evs in event_steps_dict.items(): 
            wf_number = base_wf
            wf_number = wf_number + offset_era * (e_n + len(eras_2022_1))
            wf_number = wf_number + offset_pd * (p_n + len(pds_2022_1))
            wf_number = wf_number + offset_events * evs 
            wf_number = round(wf_number,6)
            step_name = 'Run' + pd + era.split('Run')[1] + '_' + e_key
            y = str(int(base_wf))
            suff = 'ZB_' if 'ZeroBias' in step_name else ''
            workflows[wf_number] = ['',[step_name,'HLTDR3_' + y,'RECONANORUN3_' + suff + 'reHLT_'+y,'HARVESTRUN3_' + suff + y]] 
