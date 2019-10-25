from copy import deepcopy
from collections import OrderedDict
import six
from .MatrixUtil import merge

# DON'T CHANGE THE ORDER, only append new keys. Otherwise the numbering for the runTheMatrix tests will change.

upgradeKeys = {}

upgradeKeys[2017] = [
    '2017',
    '2017PU',
    '2017Design',
    '2017DesignPU',
    '2018',
    '2018PU',
    '2018Design',
    '2018DesignPU',
    '2021',
    '2021PU',
    '2021Design',
    '2021DesignPU',
    '2023',
    '2023PU',
    '2024',
    '2024PU',
]

upgradeKeys[2026] = [
    '2026D35',
    '2026D35PU',
    '2026D41',
    '2026D41PU',
    '2026D43',
    '2026D43PU',
    '2026D44',
    '2026D44PU',
    '2026D45',
    '2026D45PU',
    '2026D46',
    '2026D46PU',
    '2026D47',
    '2026D47PU',
    '2026D48',
    '2026D48PU',
    '2026D49',
    '2026D49PU',
]

# pre-generation of WF numbers
numWFStart={
    2017: 10000,
    2026: 20000,
}
numWFSkip=200
# temporary measure to keep other WF numbers the same
numWFConflict = [[25000,26000],[50000,51000]]
numWFAll={
    2017: [],
    2026: []
}

for year in upgradeKeys:
    for i in range(0,len(upgradeKeys[year])):
        numWFtmp = numWFStart[year] if i==0 else (numWFAll[year][i-1] + numWFSkip)
        for conflict in numWFConflict:
            if numWFtmp>=conflict[0] and numWFtmp<conflict[1]:
                numWFtmp = conflict[1]
                break
        numWFAll[year].append(numWFtmp)

# workflows for baseline and for variations
# setup() automatically loops over all steps and applies any customizations specified in setup_() -> called in relval_steps.py
# workflow() adds a concrete workflow to the list based on condition() -> called in relval_upgrade.py
# every special workflow gets its own derived class, which must then be added to the global dict upgradeWFs
class UpgradeWorkflow(object):
    def __init__(self,steps,PU,suffix,offset):
        self.steps = steps
        self.PU = PU
        self.suffix = suffix
        self.offset = offset
        if self.offset < 0.0 or self.offset > 1.0:
            raise ValueError("Special workflow offset must be between 0.0 and 1.0")
    def init(self, stepDict):
        for step in self.steps:
            stepName = step + self.suffix
            stepDict[stepName] = {}
        for step in self.PU:
            stepName = step + 'PU' + self.suffix
            stepDict[stepName] = {}
            stepNamePmx = step + 'PUPRMX' + self.suffix
            stepDict[stepNamePmx] = {}
            stepDict[stepNamePmx+'Combined'] = {}
    def setup(self, stepDict, k, properties):
        for step in self.steps:
            stepName = step + self.suffix
            self.setup_(step, stepName, stepDict, k, properties)
    def setup_(self, step, stepName, stepDict, k, properties):
        pass
    def workflow(self, workflows, num, fragment, stepList, key, hasHarvest):
        if self.condition(fragment, stepList, key, hasHarvest):
            self.workflow_(workflows, num, fragment, stepList)
    def workflow_(self, workflows, num, fragment, stepList):
        workflows[num+self.offset] = [ fragment, stepList ]
    def condition(self, fragment, stepList, key, hasHarvest):
        return False
upgradeWFs = OrderedDict()

class UpgradeWorkflow_baseline(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        cust=properties.get('Custom', None)
        era=properties.get('Era', None)
        if cust is not None: stepDict[stepName][k]['--customise']=cust
        if era is not None: stepDict[stepName][k]['--era']=era
    def condition(self, fragment, stepList, key, hasHarvest):
        return True
upgradeWFs['baseline'] = UpgradeWorkflow_baseline(
    steps =  [
        'GenSimFull',
        'GenSimHLBeamSpotFull',
        'GenSimHLBeamSpotFull14',
        'DigiFull',
        'DigiFullTrigger',
        'RecoFullLocal',
        'RecoFull',
        'RecoFullGlobal',
        'HARVESTFull',
        'FastSim',
        'HARVESTFast',
        'HARVESTFullGlobal',
        'ALCAFull',
        'NanoFull',
        'MiniAODFullGlobal',
    ],
    PU =  [
        'DigiFullTrigger',
        'RecoFullLocal',
        'RecoFullGlobal',
        'DigiFull',
        'RecoFull',
        'HARVESTFull',
        'HARVESTFullGlobal',
        'MiniAODFullGlobal',
        'NanoFull',
    ],
    suffix = '',
    offset = 0.0,
)

# some commonalities among tracking WFs
class UpgradeWorkflowTracking(UpgradeWorkflow):
    def condition(self, fragment, stepList, key, hasHarvest):
        result = (fragment=="TTbar_13" or fragment=="TTbar_14TeV") and not 'PU' in key and hasHarvest and self.condition_(fragment, stepList, key, hasHarvest)
        if result:
            # skip ALCA and Nano
            stepList = [s for s in stepList if (("ALCA" not in s) and ("Nano" not in s))]
        return result
    def condition_(self, fragment, stepList, key, hasHarvest):
        return True

class UpgradeWorkflow_trackingOnly(UpgradeWorkflowTracking):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step: stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM'}, stepDict[step][k]])
upgradeWFs['trackingOnly'] = UpgradeWorkflow_trackingOnly(
    steps = [
        'RecoFull',
        'HARVESTFull',
        'RecoFullGlobal',
        'HARVESTFullGlobal',
    ],
    PU = [],
    suffix = '_trackingOnly',
    offset = 0.1,
)
upgradeWFs['trackingOnly'].step3 = {
    '-s': 'RAW2DIGI,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM',
    '--datatier':'GEN-SIM-RECO,DQMIO',
    '--eventcontent':'RECOSIM,DQM',
}
# used outside of upgrade WFs
step3_trackingOnly = upgradeWFs['trackingOnly'].step3

class UpgradeWorkflow_trackingRun2(UpgradeWorkflowTracking):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step and stepDict[step][k]['--era']=='Run2_2017':
            stepDict[stepName][k] = merge([{'--era': 'Run2_2017_trackingRun2'}, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2017' in key
upgradeWFs['trackingRun2'] = UpgradeWorkflow_trackingRun2(
    steps = [
        'RecoFull',
    ],
    PU = [],
    suffix = '_trackingRun2',
    offset = 0.2,
)

class UpgradeWorkflow_trackingOnlyRun2(UpgradeWorkflowTracking):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step and stepDict[step][k]['--era']=='Run2_2017':
            stepDict[stepName][k] = merge([{'--era': 'Run2_2017_trackingRun2'}, self.step3, stepDict[step][k]])
        elif 'HARVEST' in step: stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM'}, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2017' in key
upgradeWFs['trackingOnlyRun2'] = UpgradeWorkflow_trackingOnlyRun2(
    steps = [
        'RecoFull',
        'HARVESTFull',
    ],
    PU = [],
    suffix = '_trackingOnlyRun2',
    offset = 0.3,
)
upgradeWFs['trackingOnlyRun2'].step3 = upgradeWFs['trackingOnly'].step3

class UpgradeWorkflow_trackingLowPU(UpgradeWorkflowTracking):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step and stepDict[step][k]['--era']=='Run2_2017':
            stepDict[stepName][k] = merge([{'--era': 'Run2_2017_trackingLowPU'}, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2017' in key
upgradeWFs['trackingLowPU'] = UpgradeWorkflow_trackingLowPU(
    steps = [
        'RecoFull',
    ],
    PU = [],
    suffix = '_trackingLowPU',
    offset = 0.4,
)

class UpgradeWorkflow_pixelTrackingOnly(UpgradeWorkflowTracking):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step: stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'}, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2017' in key or '2018' in key
upgradeWFs['pixelTrackingOnly'] = UpgradeWorkflow_pixelTrackingOnly(
    steps = [
        'RecoFull',
        'HARVESTFull',
        'RecoFullGlobal',
        'HARVESTFullGlobal',
    ],
    PU = [],
    suffix = '_pixelTrackingOnly',
    offset = 0.5,
)
upgradeWFs['pixelTrackingOnly'].step3 = {
    '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
    '--datatier': 'GEN-SIM-RECO,DQMIO',
    '--eventcontent': 'RECOSIM,DQM',
}

class UpgradeWorkflow_trackingMkFit(UpgradeWorkflowTracking):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2017' in key or '2021' in key
upgradeWFs['trackingMkFit'] = UpgradeWorkflow_trackingMkFit(
    steps = [
        'RecoFull',
        'RecoFullGlobal',
    ],
    PU = [],
    suffix = '_trackingMkFit',
    offset = 0.7,
)
upgradeWFs['trackingMkFit'].step3 = {
    '--customise': 'RecoTracker/MkFit/customizeInitialStepToMkFit.customizeInitialStepToMkFit'
}

class UpgradeWorkflow_ProdLike(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step:
            stepDict[stepName][k] = merge([{'-s': 'RAW2DIGI,L1Reco,RECO,RECOSIM', '--datatier':'GEN-SIM-RECO', '--eventcontent':'FEVTDEBUGHLT'}, stepDict[step][k]])
        elif 'MiniAOD' in step:
            # the separate miniAOD step is used here
            stepDict[stepName][k] = deepcopy(stepDict[step][k])
        if 'HARVEST' in step:
            # remove step
            stepDict[stepName][k] = None
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and '2026' in key
upgradeWFs['ProdLike'] = UpgradeWorkflow_ProdLike(
    steps = [
        'RecoFullGlobal',
        'HARVESTFullGlobal',
        'MiniAODFullGlobal',
    ],
    PU = [
        'RecoFullGlobal',
        'HARVESTFullGlobal',
        'MiniAODFullGlobal',
    ],
    suffix = '_ProdLike',
    offset = 0.21,
)

class UpgradeWorkflow_Neutron(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'GenSim' in step:
            custNew = "SimG4Core/Application/NeutronBGforMuonsXS_cff.customise"
        else:
            custNew = "SLHCUpgradeSimulations/Configuration/customise_mixing.customise_Mix_LongLived_Neutrons"
        stepDict[stepName][k] = deepcopy(stepDict[step][k])
        if '--customise' in stepDict[stepName][k].keys():
            stepDict[stepName][k]['--customise'] += ","+custNew
        else:
            stepDict[stepName][k]['--customise'] = custNew
    def condition(self, fragment, stepList, key, hasHarvest):
        return any(fragment==nfrag for nfrag in self.neutronFrags) and any(nkey in key for nkey in self.neutronKeys)
upgradeWFs['Neutron'] = UpgradeWorkflow_Neutron(
    steps = [
        'GenSimFull',
        'GenSimHLBeamSpotFull',
        'GenSimHLBeamSpotFull14',
        'DigiFull',
        'DigiFullTrigger',
    ],
    PU = [
        'DigiFull',
        'DigiFullTrigger',
    ],
    suffix = '_Neutron',
    offset = 0.12,
)
# add some extra info
upgradeWFs['Neutron'].neutronKeys = [x for x in upgradeKeys[2026] if 'PU' not in x]
upgradeWFs['Neutron'].neutronFrags = ['ZMM_14','MinBias_14TeV']

class UpgradeWorkflow_heCollapse(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = merge([{'--procModifiers': 'run2_HECollapse_2018'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_13" and '2018' in key
upgradeWFs['heCollapse'] = UpgradeWorkflow_heCollapse(
    steps = [
        'GenSimFull',
        'DigiFull',
        'RecoFull',
        'HARVESTFull',
        'ALCAFull',
    ],
    PU = [
        'DigiFull',
        'RecoFull',
        'HARVESTFull',
    ],
    suffix = '_heCollapse',
    offset = 0.6,
)

class UpgradeWorkflow_ParkingBPH(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step and 'Run2_2018' in stepDict[step][k]['--era']:
            stepDict[stepName][k] = merge([{'--era': 'Run2_2018,bParking'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_13" and '2018' in key
upgradeWFs['ParkingBPH'] = UpgradeWorkflow_ParkingBPH(
    steps = [
        'RecoFull',
    ],
    PU = [],
    suffix = '_ParkingBPH',
    offset = 0.8,
)

class UpgradeWorkflow_TICLOnly(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        result = (fragment=="CloseByParticleGun") and ('2026' in key)
        if result: stepList = [s for s in stepList if ("HARVEST" not in s)]
        return result
upgradeWFs['TICLOnly'] = UpgradeWorkflow_TICLOnly(
    steps = [
        'RecoFull',
        'RecoFullGlobal',
    ],
    PU = [],
    suffix = '_TICLOnly',
    offset = 0.51,
)
upgradeWFs['TICLOnly'].step3 = {
    '--customise' : 'RecoHGCal/TICL/ticl_iterations.TICL_iterations'
}

class UpgradeWorkflow_TICLFullReco(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="CloseByParticleGun") and ('2026' in key)
upgradeWFs['TICLFullReco'] = UpgradeWorkflow_TICLFullReco(
    steps = [
        'RecoFull',
        'RecoFullGlobal',
    ],
    PU = [],
    suffix = '_TICLFullReco',
    offset = 0.52,
)
upgradeWFs['TICLFullReco'].step3 = {
    '--customise' : 'RecoHGCal/TICL/ticl_iterations.TICL_iterations_withReco'
}

# for premix, just use base class to store information
# actual operations happen in relval_steps.py and relval_upgrade.py
upgradeWFs['Premix'] = UpgradeWorkflow(
    steps = [],
    PU = [
        'PremixFull',
        'PremixHLBeamSpotFull',
        'PremixHLBeamSpotFull14',
    ],
    suffix = '_Premix',
    offset = 0.97,
)
# Premix stage2 is derived from baseline+PU in relval_upgrade.py
upgradeWFs['premixS2'] = UpgradeWorkflow(
    steps = [],
    PU = [],
    suffix = '_premixS2',
    offset = 0.98,
)
# Premix combined stage1+stage2 is derived for Premix+PU and baseline+PU in relval_upgrade.py
upgradeWFs['premixS1S2'] = UpgradeWorkflow(
    steps = [],
    PU = [],
    suffix = '_premixS1S2',
    offset = 0.99,
)

# check for duplicate offsets
offsets = [specialWF.offset for specialType,specialWF in six.iteritems(upgradeWFs)]
seen = set()
dups = set(x for x in offsets if x in seen or seen.add(x))
if len(dups)>0:
    raise ValueError("Duplicate special workflow offsets not allowed: "+','.join(list(dups)))

upgradeProperties = {}

upgradeProperties[2017] = {
    '2017' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2017_realistic',
        'HLTmenu': '@relval2017',
        'Era' : 'Run2_2017',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull','ALCAFull','NanoFull'],
    },
    '2017Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2017_design',
        'HLTmenu': '@relval2017',
        'Era' : 'Run2_2017',
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
    },
    '2018' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2018_realistic',
        'HLTmenu': '@relval2018',
        'Era' : 'Run2_2018',
        'BeamSpot': 'Realistic25ns13TeVEarly2018Collision',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull','ALCAFull','NanoFull'],
    },
    '2018Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2018_design',
        'HLTmenu': '@relval2018',
        'Era' : 'Run2_2018',
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
    },
    '2021' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2021_realistic',
        'HLTmenu': '@relval2021',
        'Era' : 'Run3',
        'BeamSpot': 'Run3RoundOptics25ns13TeVLowSigmaZ',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull','ALCAFull'],
    },
    '2021Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2021_design',
        'HLTmenu': '@relval2021',
        'Era' : 'Run3',
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
    },
    '2023' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2023_realistic',
        'HLTmenu': '@relval2021',
        'Era' : 'Run3',
        'BeamSpot': 'Run3RoundOptics25ns13TeVLowSigmaZ',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull','ALCAFull'],
    },
    '2024' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2024_realistic',
        'HLTmenu': '@relval2021',
        'Era' : 'Run3',
        'BeamSpot': 'Run3RoundOptics25ns13TeVLowSigmaZ',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull','ALCAFull'],
    },
}

# standard PU sequences
for key in list(upgradeProperties[2017].keys()):
    upgradeProperties[2017][key+'PU'] = deepcopy(upgradeProperties[2017][key])
    upgradeProperties[2017][key+'PU']['ScenToRun'] = ['GenSimFull','DigiFullPU','RecoFullPU','HARVESTFullPU'] + \
                                                     ['NanoFull'] if 'design' not in key else []

upgradeProperties[2026] = {
    '2026D35' : {
        'Geom' : 'Extended2026D35',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T6',
        'Era' : 'Phase2C4_timing_layer_bar',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2026D41' : {
        'Geom' : 'Extended2026D41',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T14',
        'Era' : 'Phase2C8_timing_layer_bar',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2026D43' : {
        'Geom' : 'Extended2026D43',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T14',
        'Era' : 'Phase2C4_timing_layer_bar',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2026D44' : {
        'Geom' : 'Extended2026D44',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T14',
        'Era' : 'Phase2C6_timing_layer_bar',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2026D45' : {
        'Geom' : 'Extended2026D45',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T15',
        'Era' : 'Phase2C8_timing_layer_bar',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2026D46' : {
        'Geom' : 'Extended2026D46',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T15',
        'Era' : 'Phase2C9_timing_layer_bar',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2026D47' : {
        'Geom' : 'Extended2026D47',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T15',
        'Era' : 'Phase2C10_timing_layer_bar',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2026D48' : {
        'Geom' : 'Extended2026D48',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T15',
        'Era' : 'Phase2C9_timing_layer_bar',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2026D49' : {
        'Geom' : 'Extended2026D49',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T15',
        'Era' : 'Phase2C9_timing_layer_bar',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
}

# standard PU sequences
for key in list(upgradeProperties[2026].keys()):
    upgradeProperties[2026][key+'PU'] = deepcopy(upgradeProperties[2026][key])
    upgradeProperties[2026][key+'PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']

# for relvals
defaultDataSets = {}
for year in upgradeKeys:
    for key in upgradeKeys[year]:
        if 'PU' in key: continue
        defaultDataSets[key] = ''

from  Configuration.PyReleaseValidation.relval_steps import Kby

upgradeFragments=['FourMuPt_1_200_pythia8_cfi',
                  'SingleElectronPt10_pythia8_cfi',
                  'SingleElectronPt35_pythia8_cfi',
                  'SingleElectronPt1000_pythia8_cfi',
                  'SingleGammaPt10_pythia8_cfi',
                  'SingleGammaPt35_pythia8_cfi',
                  'SingleMuPt1_pythia8_cfi',
                  'SingleMuPt10_pythia8_cfi',
                  'SingleMuPt100_pythia8_cfi',
                  'SingleMuPt1000_pythia8_cfi',
                  'FourMuExtendedPt_1_200_pythia8_cfi',
                  'TenMuExtendedE_0_200_pythia8_cfi',
                  'DoubleElectronPt10Extended_pythia8_cfi',
                  'DoubleElectronPt35Extended_pythia8_cfi',
                  'DoubleElectronPt1000Extended_pythia8_cfi',
                  'DoubleGammaPt10Extended_pythia8_cfi',
                  'DoubleGammaPt35Extended_pythia8_cfi',
                  'DoubleMuPt1Extended_pythia8_cfi',
                  'DoubleMuPt10Extended_pythia8_cfi',
                  'DoubleMuPt100Extended_pythia8_cfi',
                  'DoubleMuPt1000Extended_pythia8_cfi',
                  'TenMuE_0_200_pythia8_cfi',
                  'SinglePiE50HCAL_pythia8_cfi',
                  'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi', 
                  'TTbar_13TeV_TuneCUETP8M1_cfi',
                  'ZEE_13TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt_600_800_13TeV_TuneCUETP8M1_cfi',
                  'Wjet_Pt_80_120_14TeV_TuneCUETP8M1_cfi',
                  'Wjet_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi',
                  'LM1_sfts_14TeV_cfi',
                  'QCD_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt_80_120_14TeV_TuneCUETP8M1_cfi',
                  'H200ChargedTaus_Tauola_14TeV_cfi',
                  'JpsiMM_14TeV_TuneCUETP8M1_cfi',
                  'TTbar_14TeV_TuneCUETP8M1_cfi',
                  'WE_14TeV_TuneCUETP8M1_cfi',
                  'ZTT_Tauola_All_hadronic_14TeV_TuneCUETP8M1_cfi',
                  'H130GGgluonfusion_14TeV_TuneCUETP8M1_cfi',
                  'PhotonJet_Pt_10_14TeV_TuneCUETP8M1_cfi',
                  'QQH1352T_Tauola_14TeV_TuneCUETP8M1_cfi',
                  'MinBias_14TeV_pythia8_TuneCUETP8M1_cfi',
                  'WM_14TeV_TuneCUETP8M1_cfi',
                  'ZMM_13TeV_TuneCUETP8M1_cfi',
                  'QCDForPF_14TeV_TuneCUETP8M1_cfi',
                  'DYToLL_M-50_14TeV_pythia8_cff',
                  'DYToTauTau_M-50_14TeV_pythia8_tauola_cff',
                  'ZEE_14TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi',
                  'H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt-20toInf_MuEnrichedPt15_TuneCUETP8M1_14TeV_pythia8_cff',
                  'ZMM_14TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt-15To7000_TuneCUETP8M1_Flat_14TeV-pythia8_cff',
                  'H125GGgluonfusion_14TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt_600_800_14TeV_TuneCUETP8M1_cfi',
                  'UndergroundCosmicSPLooseMu_cfi',
                  'BeamHalo_13TeV_cfi',
                  'H200ChargedTaus_Tauola_13TeV_cfi',
                  'ADDMonoJet_13TeV_d3MD3_TuneCUETP8M1_cfi',
                  'ZpMM_13TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi',
                  'WpM_13TeV_TuneCUETP8M1_cfi',
                  'SingleNuE10_cfi.py',
                  'TTbarLepton_13TeV_TuneCUETP8M1_cfi',
                  'WE_13TeV_TuneCUETP8M1_cfi',
                  'WM_13TeV_TuneCUETP8M1_cfi',
                  'ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi',
                  'PhotonJet_Pt_10_13TeV_TuneCUETP8M1_cfi',
                  'QQH1352T_13TeV_TuneCUETP8M1_cfi',
                  'Wjet_Pt_80_120_13TeV_TuneCUETP8M1_cfi',
                  'Wjet_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi',
                  'SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi',
                  'QCDForPF_13TeV_TuneCUETP8M1_cfi',
                  'PYTHIA8_PhiToMuMu_TuneCUETP8M1_13TeV_cff',
                  'RSKKGluon_m3000GeV_13TeV_TuneCUETP8M1_cff',
                  'ZpMM_2250_13TeV_TuneCUETP8M1_cfi',
                  'ZpEE_2250_13TeV_TuneCUETP8M1_cfi',
                  'ZpTT_1500_13TeV_TuneCUETP8M1_cfi',
                  'Upsilon1SToMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi',
                  'EtaBToJpsiJpsi_forSTEAM_TuneCUEP8M1_13TeV_cfi',
                  'JpsiMuMu_Pt-8_forSTEAM_13TeV_TuneCUETP8M1_cfi',
                  'BuMixing_BMuonFilter_forSTEAM_13TeV_TuneCUETP8M1_cfi',
                  'HSCPstop_M_200_TuneCUETP8M1_13TeV_pythia8_cff',
                  'RSGravitonToGammaGamma_kMpl01_M_3000_TuneCUETP8M1_13TeV_pythia8_cfi',
                  'WprimeToENu_M-2000_TuneCUETP8M1_13TeV-pythia8_cff',
                  'DisplacedSUSY_stopToBottom_M_300_1000mm_TuneCUETP8M1_13TeV_pythia8_cff',
                  'TenE_E_0_200_pythia8_cfi',
                  'FlatRandomPtAndDxyGunProducer_cfi',
                  'TenTau_E_15_500_pythia8_cfi',
                  'SinglePiPt25Eta1p7_2p7_cfi',
                  'SingleMuPt15Eta1p7_2p7_cfi',
                  'SingleGammaPt25Eta1p7_2p7_cfi',
                  'SingleElectronPt15Eta1p7_2p7_cfi',
                  'ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi',
                  'CloseByParticle_Photon_ERZRanges_cfi',
                  'CE_E_Front_300um_cfi', 
                  'CE_E_Front_200um_cfi', 
                  'CE_E_Front_120um_cfi', 
                  'CE_H_Fine_300um_cfi',  
                  'CE_H_Fine_200um_cfi',  
                  'CE_H_Fine_120um_cfi',  
                  'CE_H_Coarse_Scint_cfi',
                  'CE_H_Coarse_300um_cfi',
]

howMuches={'FourMuPt_1_200_pythia8_cfi':Kby(10,100),
           'TenMuE_0_200_pythia8_cfi':Kby(10,100),
           'FourMuExtendedPt_1_200_pythia8_cfi':Kby(10,100),
           'TenMuExtendedE_0_200_pythia8_cfi':Kby(10,100),
           'SingleElectronPt10_pythia8_cfi':Kby(9,100),
           'SingleElectronPt35_pythia8_cfi':Kby(9,100),
           'SingleElectronPt1000_pythia8_cfi':Kby(9,50),
           'SingleGammaPt10_pythia8_cfi':Kby(9,100),
           'SingleGammaPt35_pythia8_cfi':Kby(9,50),
           'SingleMuPt1_pythia8_cfi':Kby(25,100),
           'SingleMuPt10_pythia8_cfi':Kby(25,100),
           'SingleMuPt100_pythia8_cfi':Kby(9,100),
           'SingleMuPt1000_pythia8_cfi':Kby(9,100),
           'DoubleElectronPt10Extended_pythia8_cfi':Kby(9,100),
           'DoubleElectronPt35Extended_pythia8_cfi':Kby(9,100),
           'DoubleElectronPt1000Extended_pythia8_cfi':Kby(9,50),
           'DoubleGammaPt10Extended_pythia8_cfi':Kby(9,100),
           'DoubleGammaPt35Extended_pythia8_cfi':Kby(9,50),
           'DoubleMuPt1Extended_pythia8_cfi':Kby(25,100),
           'DoubleMuPt10Extended_pythia8_cfi':Kby(25,100),
           'DoubleMuPt100Extended_pythia8_cfi':Kby(9,100),
           'DoubleMuPt1000Extended_pythia8_cfi':Kby(9,100),
           'SinglePiE50HCAL_pythia8_cfi':Kby(50,500),
           'QCD_Pt_600_800_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'Wjet_Pt_80_120_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'Wjet_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'LM1_sfts_14TeV_cfi':Kby(9,100),
           'QCD_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'QCD_Pt_80_120_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'H200ChargedTaus_Tauola_14TeV_cfi':Kby(9,100),
           'JpsiMM_14TeV_TuneCUETP8M1_cfi':Kby(66,100),
           'TTbar_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'WE_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'ZEE_13TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'ZTT_Tauola_All_hadronic_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'H130GGgluonfusion_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'PhotonJet_Pt_10_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'QQH1352T_Tauola_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'MinBias_14TeV_pythia8_TuneCUETP8M1_cfi':Kby(90,100),
           'WM_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'ZMM_13TeV_TuneCUETP8M1_cfi':Kby(18,100),
           'QCDForPF_14TeV_TuneCUETP8M1_cfi':Kby(50,100),
           'DYToLL_M-50_14TeV_pythia8_cff':Kby(9,100),
           'DYToTauTau_M-50_14TeV_pythia8_tauola_cff':Kby(9,100),
           'TTbar_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi':Kby(90,100),
           'ZEE_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'QCD_Pt-20toInf_MuEnrichedPt15_TuneCUETP8M1_14TeV_pythia8_cff':Kby(9,100),
           'ZMM_14TeV_TuneCUETP8M1_cfi':Kby(18,100),
           'QCD_Pt-15To7000_TuneCUETP8M1_Flat_14TeV-pythia8_cff':Kby(9,50),
           'H125GGgluonfusion_14TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'QCD_Pt_600_800_14TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'UndergroundCosmicSPLooseMu_cfi':Kby(9,50),
           'BeamHalo_13TeV_cfi':Kby(9,50),
           'H200ChargedTaus_Tauola_13TeV_cfi':Kby(9,50),
           'ADDMonoJet_13TeV_d3MD3_TuneCUETP8M1_cfi':Kby(9,50),
           'ZpMM_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'WpM_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'SingleNuE10_cfi.py':Kby(9,50),
           'TTbarLepton_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'WE_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'WM_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'PhotonJet_Pt_10_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'QQH1352T_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'Wjet_Pt_80_120_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'Wjet_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi':Kby(9,50),
           'QCDForPF_13TeV_TuneCUETP8M1_cfi':Kby(50,100),
           'PYTHIA8_PhiToMuMu_TuneCUETP8M1_13TeV_cff':Kby(9,50),
           'RSKKGluon_m3000GeV_13TeV_TuneCUETP8M1_cff':Kby(9,50),
           'ZpMM_2250_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'ZpEE_2250_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'ZpTT_1500_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'Upsilon1SToMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'EtaBToJpsiJpsi_forSTEAM_TuneCUEP8M1_13TeV_cfi':Kby(9,50),
           'JpsiMuMu_Pt-8_forSTEAM_13TeV_TuneCUETP8M1_cfi':Kby(3100,100000),
           'BuMixing_BMuonFilter_forSTEAM_13TeV_TuneCUETP8M1_cfi':Kby(900,10000),
           'HSCPstop_M_200_TuneCUETP8M1_13TeV_pythia8_cff':Kby(9,50),
           'RSGravitonToGammaGamma_kMpl01_M_3000_TuneCUETP8M1_13TeV_pythia8_cfi':Kby(9,50),
           'WprimeToENu_M-2000_TuneCUETP8M1_13TeV-pythia8_cff':Kby(9,50),
           'DisplacedSUSY_stopToBottom_M_300_1000mm_TuneCUETP8M1_13TeV_pythia8_cff':Kby(9,50),
           'TenE_E_0_200_pythia8_cfi':Kby(9,100),
           'FlatRandomPtAndDxyGunProducer_cfi':Kby(9,100),
           'TenTau_E_15_500_pythia8_cfi':Kby(9,100),
           'SinglePiPt25Eta1p7_2p7_cfi':Kby(9,100),
           'SingleMuPt15Eta1p7_2p7_cfi':Kby(9,100),
           'SingleGammaPt25Eta1p7_2p7_cfi':Kby(9,100),
           'SingleElectronPt15Eta1p7_2p7_cfi':Kby(9,100),
           'ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'CloseByParticle_Photon_ERZRanges_cfi':Kby(9,100),
           'CE_E_Front_300um_cfi':Kby(9,100), 
           'CE_E_Front_200um_cfi':Kby(9,100), 
           'CE_E_Front_120um_cfi':Kby(9,100), 
           'CE_H_Fine_300um_cfi':Kby(9,100),  
           'CE_H_Fine_200um_cfi':Kby(9,100),  
           'CE_H_Fine_120um_cfi':Kby(9,100),  
           'CE_H_Coarse_Scint_cfi':Kby(9,100),
           'CE_H_Coarse_300um_cfi':Kby(9,100),
}

upgradeDatasetFromFragment={'FourMuPt_1_200_pythia8_cfi': 'FourMuPt1_200',
                            'FourMuExtendedPt_1_200_pythia8_cfi': 'FourMuExtendedPt1_200',
                            'TenMuE_0_200_pythia8_cfi': 'TenMuE_0_200',
                            'TenMuExtendedE_0_200_pythia8_cfi': 'TenMuExtendedE_0_200',
                            'SingleElectronPt10_pythia8_cfi' : 'SingleElectronPt10',
                            'SingleElectronPt35_pythia8_cfi' : 'SingleElectronPt35',
                            'SingleElectronPt1000_pythia8_cfi' : 'SingleElectronPt1000',
                            'SingleGammaPt10_pythia8_cfi' : 'SingleGammaPt10',
                            'SingleGammaPt35_pythia8_cfi' : 'SingleGammaPt35',
                            'SingleMuPt1_pythia8_cfi' : 'SingleMuPt1',
                            'SingleMuPt10_pythia8_cfi' : 'SingleMuPt10',
                            'SingleMuPt100_pythia8_cfi' : 'SingleMuPt100',
                            'SingleMuPt1000_pythia8_cfi' : 'SingleMuPt1000',
                            'DoubleElectronPt10Extended_pythia8_cfi' : 'SingleElectronPt10Extended',
                            'DoubleElectronPt35Extended_pythia8_cfi' : 'SingleElectronPt35Extended',
                            'DoubleElectronPt1000Extended_pythia8_cfi' : 'SingleElectronPt1000Extended',
                            'DoubleGammaPt10Extended_pythia8_cfi' : 'SingleGammaPt10Extended',
                            'DoubleGammaPt35Extended_pythia8_cfi' : 'SingleGammaPt35Extended',
                            'DoubleMuPt1Extended_pythia8_cfi' : 'SingleMuPt1Extended',
                            'DoubleMuPt10Extended_pythia8_cfi' : 'SingleMuPt10Extended',
                            'DoubleMuPt100Extended_pythia8_cfi' : 'SingleMuPt100Extended',
                            'DoubleMuPt1000Extended_pythia8_cfi' : 'SingleMuPt1000Extended',
                            'SinglePiE50HCAL_pythia8_cfi' : 'SinglePiE50HCAL',
                            'QCD_Pt_600_800_13TeV_TuneCUETP8M1_cfi' : 'QCD_Pt_600_800_13',
                            'Wjet_Pt_80_120_14TeV_TuneCUETP8M1_cfi' : 'Wjet_Pt_80_120_14TeV',
                            'Wjet_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi' : 'Wjet_Pt_3000_3500_14TeV',
                            'LM1_sfts_14TeV_cfi' : 'LM1_sfts_14TeV',
                            'QCD_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi' : 'QCD_Pt_3000_3500_14TeV',
                            'QCD_Pt_80_120_14TeV_TuneCUETP8M1_cfi' : 'QCD_Pt_80_120_14TeV',
                            'H200ChargedTaus_Tauola_14TeV_cfi' : 'Higgs200ChargedTaus_14TeV',
                            'JpsiMM_14TeV_TuneCUETP8M1_cfi' : 'JpsiMM_14TeV',
                            'TTbar_14TeV_TuneCUETP8M1_cfi' : 'TTbar_14TeV',
                            'WE_14TeV_TuneCUETP8M1_cfi' : 'WE_14TeV',
                            'ZEE_13TeV_TuneCUETP8M1_cfi' : 'ZEE_13',
                            'ZTT_Tauola_All_hadronic_14TeV_TuneCUETP8M1_cfi' : 'ZTT_14TeV',
                            'H130GGgluonfusion_14TeV_TuneCUETP8M1_cfi' : 'H130GGgluonfusion_14TeV',
                            'PhotonJet_Pt_10_14TeV_TuneCUETP8M1_cfi' : 'PhotonJets_Pt_10_14TeV',
                            'QQH1352T_Tauola_14TeV_TuneCUETP8M1_cfi' : 'QQH1352T_Tauola_14TeV',
                            'MinBias_14TeV_pythia8_TuneCUETP8M1_cfi' : 'MinBias_14TeV',
                            'WM_14TeV_TuneCUETP8M1_cfi' : 'WM_14TeV',
                            'ZMM_13TeV_TuneCUETP8M1_cfi' : 'ZMM_13',
                            'QCDForPF_14TeV_TuneCUETP8M1_cfi' : 'QCD_FlatPt_15_3000HS_14',
                            'DYToLL_M-50_14TeV_pythia8_cff' : 'DYToLL_M_50_14TeV',
                            'DYToTauTau_M-50_14TeV_pythia8_tauola_cff' : 'DYtoTauTau_M_50_14TeV',
                            'TTbar_13TeV_TuneCUETP8M1_cfi' : 'TTbar_13',
                            'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi' : 'MinBias_13',
                            'ZEE_14TeV_TuneCUETP8M1_cfi' : 'ZEE_14',
                            'QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi' : 'QCD_Pt_80_120_13',
                            'H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi' : 'H125GGgluonfusion_13',
                            'QCD_Pt-20toInf_MuEnrichedPt15_TuneCUETP8M1_14TeV_pythia8_cff' : 'QCD_Pt-20toInf_MuEnrichedPt15_14TeV',
                            'ZMM_14TeV_TuneCUETP8M1_cfi' : 'ZMM_14',
                            'QCD_Pt-15To7000_TuneCUETP8M1_Flat_14TeV-pythia8_cff' : 'QCD_Pt-15To7000_Flat_14TeV',
                            'H125GGgluonfusion_14TeV_TuneCUETP8M1_cfi' : 'H125GGgluonfusion_14',
                            'QCD_Pt_600_800_14TeV_TuneCUETP8M1_cfi' : 'QCD_Pt_600_800_14',
                            'UndergroundCosmicSPLooseMu_cfi': 'CosmicsSPLoose',
                            'BeamHalo_13TeV_cfi': 'BeamHalo_13',
                            'H200ChargedTaus_Tauola_13TeV_cfi': 'Higgs200ChargedTaus_13',
                            'ADDMonoJet_13TeV_d3MD3_TuneCUETP8M1_cfi': 'ADDMonoJet_d3MD3_13',
                            'ZpMM_13TeV_TuneCUETP8M1_cfi': 'ZpMM_13',
                            'QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi': 'QCD_Pt_3000_3500_13',
                            'WpM_13TeV_TuneCUETP8M1_cfi': 'WpM_13',
                            'SingleNuE10_cfi.py': 'NuGun',
                            'TTbarLepton_13TeV_TuneCUETP8M1_cfi': 'TTbarLepton_13',
                            'WE_13TeV_TuneCUETP8M1_cfi': 'WE_13',
                            'WM_13TeV_TuneCUETP8M1_cfi': 'WM_13',
                            'ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi': 'ZTT_13',
                            'PhotonJet_Pt_10_13TeV_TuneCUETP8M1_cfi': 'PhotonJets_Pt_10_13',
                            'QQH1352T_13TeV_TuneCUETP8M1_cfi': 'QQH1352T_13',
                            'Wjet_Pt_80_120_13TeV_TuneCUETP8M1_cfi': 'Wjet_Pt_80_120_13',
                            'Wjet_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi': 'Wjet_Pt_3000_3500_13',
                            'SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi': 'SMS-T1tttt_mGl-1500_mLSP-100_13',
                            'QCDForPF_13TeV_TuneCUETP8M1_cfi': 'QCD_FlatPt_15_3000HS_13',
                            'PYTHIA8_PhiToMuMu_TuneCUETP8M1_13TeV_cff': 'PhiToMuMu_13',
                            'RSKKGluon_m3000GeV_13TeV_TuneCUETP8M1_cff': 'RSKKGluon_m3000GeV_13',
                            'ZpMM_2250_13TeV_TuneCUETP8M1_cfi': 'ZpMM_2250_13',
                            'ZpEE_2250_13TeV_TuneCUETP8M1_cfi': 'ZpEE_2250_13',
                            'ZpTT_1500_13TeV_TuneCUETP8M1_cfi': 'ZpTT_1500_13',
                            'Upsilon1SToMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi': 'Upsilon1SToMuMu_13',
                            'EtaBToJpsiJpsi_forSTEAM_TuneCUEP8M1_13TeV_cfi': 'EtaBToJpsiJpsi_13',
                            'JpsiMuMu_Pt-8_forSTEAM_13TeV_TuneCUETP8M1_cfi': 'JpsiMuMu_Pt-8',
                            'BuMixing_BMuonFilter_forSTEAM_13TeV_TuneCUETP8M1_cfi': 'BuMixing_13',
                            'HSCPstop_M_200_TuneCUETP8M1_13TeV_pythia8_cff': 'HSCPstop_M_200_13',
                            'RSGravitonToGammaGamma_kMpl01_M_3000_TuneCUETP8M1_13TeV_pythia8_cfi': 'RSGravitonToGaGa_13',
                            'WprimeToENu_M-2000_TuneCUETP8M1_13TeV-pythia8_cff': 'WpToENu_M-2000_13',
                            'DisplacedSUSY_stopToBottom_M_300_1000mm_TuneCUETP8M1_13TeV_pythia8_cff': 'DisplacedSUSY_stopToBottom_M_300_1000mm_13',
                            'TenE_E_0_200_pythia8_cfi': 'TenE_0_200',
                            'FlatRandomPtAndDxyGunProducer_cfi': 'DisplacedMuonsDxy_0_500',
                            'TenTau_E_15_500_pythia8_cfi':'TenTau_15_500',
                            'SinglePiPt25Eta1p7_2p7_cfi':'SinglePiPt25Eta1p7_2p7',
                            'SingleMuPt15Eta1p7_2p7_cfi':'SingleMuPt15Eta1p7_2p7',
                            'SingleGammaPt25Eta1p7_2p7_cfi':'SingleGammaPt25Eta1p7_2p7',
                            'SingleElectronPt15Eta1p7_2p7_cfi':'SingleElectronPt15Eta1p7_2p7',
                            'ZTT_All_hadronic_14TeV_TuneCUETP8M1_cfi': 'ZTT_14',
                            'CloseByParticle_Photon_ERZRanges_cfi': 'CloseByParticleGun',
                            'CE_E_Front_300um_cfi':'CloseByParticleGun_CE_E_Front_300um', 
                            'CE_E_Front_200um_cfi':'CloseByParticleGun_CE_E_Front_200um', 
                            'CE_E_Front_120um_cfi':'CloseByParticleGun_CE_E_Front_120um', 
                            'CE_H_Fine_300um_cfi':'CloseByParticleGun_CE_H_Fine_300um',  
                            'CE_H_Fine_200um_cfi':'CloseByParticleGun_CE_H_Fine_200um',  
                            'CE_H_Fine_120um_cfi':'CloseByParticleGun_CE_H_Fine_120um',  
                            'CE_H_Coarse_Scint_cfi':'CloseByParticleGun_CE_H_Coarse_Scint',
                            'CE_H_Coarse_300um_cfi':'CloseByParticleGun_CE_H_Coarse_300um',
}
