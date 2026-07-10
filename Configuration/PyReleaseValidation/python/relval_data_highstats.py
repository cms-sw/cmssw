# import the definition of the steps and input files:
from Configuration.PyReleaseValidation.relval_steps import *
from .MatrixUtil import Matrix
from functools import partial

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

## Here we define fixed high stats data workflows
## not to be run as default. 10k, 50k, 150k, 250k, 500k or 1M events each

def run3NameMod(name):
    # ParkingDouble* PDs would end up with a too long name for the submission infrastructure
    return name.replace('ParkingDouble','Park2') 

def run3HarvMod(pd):
    ## ZeroBias, ScoutingPFMonitor and ParkingDoubleMuonLowMass 
    ## have their own HARVESTING setup
    if 'ZeroBias' in pd:
        return 'ZB_'
    elif 'ScoutingPFMonitor' in pd:
        return 'ScoutingPFMonitor_'
    elif 'ParkingDoubleMuonLowMass' in pd:
        return 'HFLAV_'
    else:
        return ''

def run3RecoMod(pd):
    ## ZeroBias and ScoutingPFMonitor have 
    ## their own RECO setup
    if 'ZeroBias' in pd:
        return 'ZB_'
    elif 'ScoutingPFMonitor' in pd:
        return 'ScoutingPFMonitor_'
    else:
        return ''
    
def run3HLTMod(pd):
     ## ScoutingPFMonitor has its own HLT setup
    if 'ScoutingPFMonitor' in pd:
        return 'ScoutingPFMonitor_'
    else:
        return ''
    
def addFixedEventsWfs(years, pds, eras, offset = 0, suffreco = None, suffhlt = None, suffharv = None, namemod = None):

    for y in years:
        for era in eras:
            for pd in pds:
                for e_key,evs in event_steps_dict.items(): 

                    wf_number = float(y) + offset_pd * pds.index(pd)
                    wf_number = wf_number + offset_era * eras.index(era)
                    wf_number = wf_number + offset
                    wf_number = round(wf_number + offset_events * evs, 6)

                    # Here we customise the steps depending on the PD name                
                    reco = suffreco(pd) if suffreco is not None else ''
                    harv = suffharv(pd) if suffharv is not None else ''
                    hlt  = suffhlt(pd) if suffhlt is not None else ''
                    name = namemod(pd) if namemod is not None else ''

                    recosetup = 'RECONANORUN3_' + reco + 'reHLT_' + y 
                    harvsetup = 'HARVESTRUN3_'  + harv + y
                    hltsetup  = 'HLTDR3_' + hlt + y

                    step_name = 'Run' + name  + y + era + '_' + e_key 
                    if namemod is not None:
                        step_name = namemod(step_name)

                    workflows[wf_number] = ['',[step_name, hltsetup, recosetup, harvsetup]]

    return wf_number - float(y) #to concatenate the offset

run3FixedWfs = partial(addFixedEventsWfs,suffreco = run3RecoMod, suffhlt = run3HLTMod, suffharv = run3HarvMod, namemod = run3NameMod)
run3FixedWfs(['2025'],pds_2025,eras_2025)
run3FixedWfs(['2024'],pds_2024,eras_2024)
run3FixedWfs(['2023'],pds_2023,eras_2023)
offset_2022 = run3FixedWfs(['2022'],pds_2022_2,eras_2022_2)
run3FixedWfs(['2022'],pds_2022_1,eras_2022_1,offset = offset_2022)
