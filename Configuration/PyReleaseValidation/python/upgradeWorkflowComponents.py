from copy import copy, deepcopy
from collections import OrderedDict
from .MatrixUtil import merge, Kby, Mby, check_dups
import re

U2000by1={'--relval': '2000,1'}

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
    '2022',
    '2022PU',
    '2022Design',
    '2022DesignPU',
    '2023',
    '2023PU',
    '2024',
    '2024PU',
    '2022FS',
    '2022FSPU',
    '2022postEE',
    '2022postEEPU',
    '2023FS',
    '2023FSPU',
    '2022HI',
    '2022HIRP', #RawPrime
    '2023HI',
    '2023HIRP', #RawPrime
    '2024HLTOnDigi',
    '2024HLTOnDigiPU',
    '2024GenOnly',
    '2024SimOnGen',
    '2024FS',
    '2024FSPU',
    '2025',
    '2025PU',
    '2025HLTOnDigi',
    '2025HLTOnDigiPU',
    '2025SimOnGen',
    '2025GenOnly',
]

upgradeKeys['Run4'] = [
    'Run4D95',
    'Run4D95PU',
    'Run4D96',
    'Run4D96PU',
    'Run4D98',
    'Run4D98PU',
    'Run4D99',
    'Run4D99PU',
    'Run4D100',
    'Run4D100PU',
    'Run4D101',
    'Run4D101PU',
    'Run4D102',
    'Run4D102PU',
    'Run4D103',
    'Run4D103PU',
    'Run4D104',
    'Run4D104PU',
    'Run4D105',
    'Run4D105PU',
    'Run4D106',
    'Run4D106PU',
    'Run4D107',
    'Run4D107PU',
    'Run4D108',
    'Run4D108PU',
    'Run4D109',
    'Run4D109PU',
    'Run4D110',
    'Run4D110PU',
    'Run4D111',
    'Run4D111PU',
    'Run4D112',
    'Run4D112PU',
    'Run4D113',
    'Run4D113PU',
    'Run4D114',
    'Run4D114PU',
    'Run4D110GenOnly',
    'Run4D110SimOnGen',
    'Run4D115',
    'Run4D115PU',
    'Run4D116',
    'Run4D116PU',
]

# pre-generation of WF numbers
numWFStart={
    2017: 10000,
    'Run4': 23600,
}
numWFSkip=200
# temporary measure to keep other WF numbers the same
numWFConflict = [[14400,14800], #2022ReReco, 2022ReRecoPU (in 12_4)
                 [24400,24800], #D97
                 [50000,51000]]
numWFAll={
    2017: [],
    'Run4': []
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
# setupPU() and setupPU_() operate similarly -> called in relval_steps.py *after* merging PUDataSets w/ regular steps
# workflow() adds a concrete workflow to the list based on condition() -> called in relval_upgrade.py
# every special workflow gets its own derived class, which must then be added to the global dict upgradeWFs
preventReuseKeyword = 'NOREUSE'
class UpgradeWorkflow(object):
    def __init__(self,steps,PU,suffix,offset):
        self.steps = steps
        self.PU = PU
        self.allowReuse = True

        # ensure all PU steps are in normal step list
        for step in self.PU:
            if not step in self.steps:
                self.steps.append(step)

        self.suffix = suffix
        if len(self.suffix)>0 and self.suffix[0]!='_': self.suffix = '_'+self.suffix
        self.offset = offset
        if self.offset < 0.0 or self.offset > 1.0:
            raise ValueError("Special workflow offset must be between 0.0 and 1.0")
    def getStepName(self, step, extra=""):
        stepName = step + self.suffix + extra
        return stepName
    def getStepNamePU(self, step, extra=""):
        stepNamePU = step + 'PU' + self.suffix + extra
        return stepNamePU
    def init(self, stepDict):
        for step in self.steps:
            stepDict[self.getStepName(step)] = {}
            if not self.allowReuse: stepDict[self.getStepName(step,preventReuseKeyword)] = {}
        for step in self.PU:
            stepDict[self.getStepNamePU(step)] = {}
            if not self.allowReuse: stepDict[self.getStepNamePU(step,preventReuseKeyword)] = {}
    def setup(self, stepDict, k, properties):
        for step in self.steps:
            self.setup_(step, self.getStepName(step), stepDict, k, properties)
            if not self.allowReuse: self.preventReuse(self.getStepName(step,preventReuseKeyword), stepDict, k)
    def setupPU(self, stepDict, k, properties):
        for step in self.PU:
            self.setupPU_(step, self.getStepNamePU(step), stepDict, k, properties)
            if not self.allowReuse: self.preventReuse(self.getStepNamePU(step,preventReuseKeyword), stepDict, k)
    def setup_(self, step, stepName, stepDict, k, properties):
        pass
    def setupPU_(self, step, stepName, stepDict, k, properties):
        pass
    def workflow(self, workflows, num, fragment, stepList, key, hasHarvest):
        if self.condition(fragment, stepList, key, hasHarvest):
            self.workflow_(workflows, num, fragment, stepList, key)
    def workflow_(self, workflows, num, fragment, stepList, key):
        fragmentTmp = [fragment, key]
        if len(self.suffix)>0: fragmentTmp.append(self.suffix)
        # avoid spurious workflows (no steps modified)
        if self.offset==0 or workflows[num][1]!=stepList:
            workflows[num+self.offset] = [ fragmentTmp, stepList ]
    def condition(self, fragment, stepList, key, hasHarvest):
        return False
    def preventReuse(self, stepName, stepDict, k):
        if "Sim" in stepName and stepName != "Sim":
            stepDict[stepName][k] = None
        if "Gen" in stepName:
            stepDict[stepName][k] = None
upgradeWFs = OrderedDict()

class UpgradeWorkflow_baseline(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
  
        cust=properties.get('Custom', None)
        era=properties.get('Era', None)
        modifier=properties.get('ProcessModifier',None)

        if cust is not None: stepDict[stepName][k]['--customise']=cust
        if era is not None:
            stepDict[stepName][k]['--era']=era
        if modifier is not None: stepDict[stepName][k]['--procModifier']=modifier
    def condition(self, fragment, stepList, key, hasHarvest):
        return True
upgradeWFs['baseline'] = UpgradeWorkflow_baseline(
    steps =  [
        'Gen',
        'GenHLBeamSpot',
        'GenHLBeamSpot14',
        'Sim',
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'GenSimHLBeamSpotHGCALCloseBy',
        'Digi',
        'DigiNoHLT',
        'DigiTrigger',
        'HLTRun3',
        'HLTOnly',
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'FastSim',
        'HARVESTFast',
        'HARVESTGlobal',
        'ALCA',
        'ALCAPhase2',
        'Nano',
        'MiniAOD',
        'HLT75e33',
        'FastSimRun3',
        'HARVESTFastRun3',
    ],
    PU =  [
        'DigiTrigger',
        'RecoLocal',
        'RecoGlobal',
        'Digi',
        'DigiNoHLT',
        'HLTOnly',
        'Reco',
        'RecoFakeHLT',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'HARVESTGlobal',
        'MiniAOD',
        'Nano',
        'HLT75e33',
        'FastSimRun3',
        'HARVESTFastRun3',
    ],
    suffix = '',
    offset = 0.0,
)


class UpgradeWorkflow_DigiNoHLT(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if stepDict[step][k] != None:
            if 'ALCA' in step:
                stepDict[stepName][k] = None
            if 'RecoNano' in step:
                stepDict[stepName][k] = merge([{'--filein': 'file:step3.root', '--secondfilein': 'file:step2.root'}, stepDict[step][k]])
            if 'Digi' in step and 'NoHLT' not in step:
                stepDict[stepName][k] = merge([{'-s': re.sub(',HLT.*', '', stepDict[step][k]['-s'])}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        if ('TTbar_14TeV' in fragment and '2022' == key):
            stepList.insert(stepList.index('Digi_DigiNoHLT_2022')+1, 'HLTRun3_2022')
        return ('TTbar_14TeV' in fragment and '2022' == key)
upgradeWFs['DigiNoHLT'] = UpgradeWorkflow_DigiNoHLT(
    steps = [
        'Digi',
        'RecoNano',
        'RecoNanoFakeHLT',
        'ALCA'
    ],
    PU = [],
    suffix = '_DigiNoHLT',
    offset = 0.601,
)

# some commonalities among tracking WFs
class UpgradeWorkflowTracking(UpgradeWorkflow):

    def __init__(self, steps, PU, suffix, offset):
        # always include some steps that will be skipped
        steps = steps + ["ALCA","Nano"]
        super().__init__(steps, PU, suffix, offset)
    def condition(self, fragment, stepList, key, hasHarvest):
        result = (fragment=="TTbar_13" or fragment=="TTbar_14TeV" or 'Hydjet' in fragment) and not 'PU' in key and hasHarvest and self.condition_(fragment, stepList, key, hasHarvest)
        return result
    def condition_(self, fragment, stepList, key, hasHarvest):
        return True
    def setup_(self, step, stepName, stepDict, k, properties):
        # skip ALCA and Nano steps (but not RecoNano or HARVESTNano for Run3)
        if 'ALCA' in step or 'Nano'==step:
            stepDict[stepName][k] = None
        self.setup__(step, stepName, stepDict, k, properties)
    # subordinate function for inherited classes
    def setup__(self, step, stepName, stepDict, k, properties):
        pass

class UpgradeWorkflow_trackingOnly(UpgradeWorkflowTracking):
    def setup__(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step: stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM'}, stepDict[step][k]])

    def condition(self, fragment, stepList, key, hasHarvest):
        result = (fragment=="TTbar_13" or fragment=="TTbar_14TeV") and hasHarvest and self.condition_(fragment, stepList, key, hasHarvest)
        return result



upgradeWFs['trackingOnly'] = UpgradeWorkflow_trackingOnly(
    steps = [
        'Reco',
        'RecoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'RecoGlobal',
        'HARVESTGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
    ],
    PU = [
        'Reco',
        'RecoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'RecoGlobal',
        'HARVESTGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
    ],


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
    def setup__(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step and stepDict[step][k]['--era']=='Run2_2017':
            stepDict[stepName][k] = merge([{'--era': 'Run2_2017_trackingRun2'}, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2017' in key
upgradeWFs['trackingRun2'] = UpgradeWorkflow_trackingRun2(
    steps = [
        'Reco',
        'RecoFakeHLT',
    ],
    PU = [],
    suffix = '_trackingRun2',
    offset = 0.2,
)

class UpgradeWorkflow_trackingOnlyRun2(UpgradeWorkflowTracking):
    def setup__(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step and stepDict[step][k]['--era']=='Run2_2017':
            stepDict[stepName][k] = merge([{'--era': 'Run2_2017_trackingRun2'}, self.step3, stepDict[step][k]])
        elif 'HARVEST' in step: stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM'}, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2017' in key
upgradeWFs['trackingOnlyRun2'] = UpgradeWorkflow_trackingOnlyRun2(
    steps = [
        'Reco',
        'RecoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
    ],
    PU = [],
    suffix = '_trackingOnlyRun2',
    offset = 0.3,
)
upgradeWFs['trackingOnlyRun2'].step3 = upgradeWFs['trackingOnly'].step3

class UpgradeWorkflow_trackingLowPU(UpgradeWorkflowTracking):
    def setup__(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step and stepDict[step][k]['--era']=='Run2_2017':
            stepDict[stepName][k] = merge([{'--era': 'Run2_2017_trackingLowPU'}, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2017' in key
upgradeWFs['trackingLowPU'] = UpgradeWorkflow_trackingLowPU(
    steps = [
        'Reco',
        'RecoFakeHLT',
    ],
    PU = [],
    suffix = '_trackingLowPU',
    offset = 0.4,
)

class UpgradeWorkflow_pixelTrackingOnly(UpgradeWorkflowTracking):
    def setup__(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        # skip ALCA step as products might not be available
        elif 'ALCA' in step: stepDict[stepName][k] = None
        elif 'HARVEST' in step: stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'}, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return ('2022' in key or '2023' in key or '2024' in key or 'Run4' in key or 'HI' in key) and ('FS' not in key)
upgradeWFs['pixelTrackingOnly'] = UpgradeWorkflow_pixelTrackingOnly(
    steps = [
        'Reco',
        'RecoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'RecoGlobal',
        'HARVESTGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'ALCA',
        'ALCAPhase2'
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
    def setup__(self, step, stepName, stepDict, k, properties):
        if ('Digi' in step and 'NoHLT' not in step) or ('HLTOnly' in step): stepDict[stepName][k] = merge([self.step2, stepDict[step][k]])
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):     
        return any(y in key for y in ['2017','2022','2023','2024','2025']) and ('FS' not in key)
upgradeWFs['trackingMkFit'] = UpgradeWorkflow_trackingMkFit(
    steps = [
        'Digi',
        'HLTOnly',
        'DigiTrigger',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
    ],
    PU = [],
    suffix = '_trackingMkFit',
    offset = 0.7,
)
upgradeWFs['trackingMkFit'].step2 = {
    '--customise': 'RecoTracker/MkFit/customizeHLTIter0ToMkFit.customizeHLTIter0ToMkFit'
}
upgradeWFs['trackingMkFit'].step3 = {
    '--procModifiers': 'trackingMkFitDevel'
}

# mkFit for phase-2 initialStep tracking
class UpgradeWorkflow_trackingMkFitPhase2(UpgradeWorkflowTracking):
    def setup__(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return ('Run4' in key)
upgradeWFs['trackingMkFitPhase2'] = UpgradeWorkflow_trackingMkFitPhase2(
    steps = [
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
    ],
    PU = [],
    suffix = '_trackingMkFitPhase2',
    offset = 0.702,
)
upgradeWFs['trackingMkFitPhase2'].step3 = {
    '--procModifiers': 'trackingMkFitCommon,trackingMkFitInitialStep'
}

# LST on CPU, initialStep+highPtTripletStep-only tracking-only
class UpgradeWorkflow_lstOnCPUIters01TrackingOnly(UpgradeWorkflowTracking):
    def setup__(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step: stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM'}, stepDict[step][k]])
        elif 'ALCA' in step: stepDict[stepName][k] = None
    def condition(self, fragment, stepList, key, hasHarvest):
        result = (fragment=="TTbar_14TeV") and hasHarvest and ('Run4' in key)
        return result
upgradeWFs['lstOnCPUIters01TrackingOnly'] = UpgradeWorkflow_lstOnCPUIters01TrackingOnly(
    steps = [
        'RecoGlobal',
        'HARVESTGlobal',
        # Add ALCA steps explicitly, so that they can be properly removed
        'ALCA',
        'ALCAPhase2'
    ],
    PU = [
        'RecoGlobal',
        'HARVESTGlobal',
    ],
    suffix = '_lstOnCPUIters01TrackingOnly',
    offset = 0.703,
)
upgradeWFs['lstOnCPUIters01TrackingOnly'].step3 = upgradeWFs['trackingOnly'].step3 | {
    '--procModifiers': 'trackingIters01,trackingLST',
    '--accelerators' : 'cpu'
}

# LST on GPU (if available), initialStep+highPtTripletStep-only tracking-only
class UpgradeWorkflow_lstOnGPUIters01TrackingOnly(UpgradeWorkflowTracking):
    def setup__(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step: stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM'}, stepDict[step][k]])
        elif 'ALCA' in step: stepDict[stepName][k] = None
    def condition(self, fragment, stepList, key, hasHarvest):
        result = (fragment=="TTbar_14TeV") and hasHarvest and ('Run4' in key)
        return result
upgradeWFs['lstOnGPUIters01TrackingOnly'] = UpgradeWorkflow_lstOnGPUIters01TrackingOnly(
    steps = [
        'RecoGlobal',
        'HARVESTGlobal',
        # Add ALCA steps explicitly, so that they can be properly removed
        'ALCA',
        'ALCAPhase2'
    ],
    PU = [
        'RecoGlobal',
        'HARVESTGlobal',
    ],
    suffix = '_lstOnGPUIters01TrackingOnly',
    offset = 0.704,
)
upgradeWFs['lstOnGPUIters01TrackingOnly'].step3 = upgradeWFs['trackingOnly'].step3 | {
    '--procModifiers': 'trackingIters01,trackingLST',
}

#DeepCore seeding for JetCore iteration workflow
class UpgradeWorkflow_seedingDeepCore(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        # skip ALCA and Nano steps (but not RecoNano or HARVESTNano for Run3)
        if 'ALCA' in step or 'Nano'==step:
            stepDict[stepName][k] = None
        elif 'Reco' in step or 'HARVEST' in step: stepDict[stepName][k] = merge([{'--procModifiers': 'seedingDeepCore'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        result = (fragment=="QCD_Pt_1800_2400_14" or fragment=="TTbar_14TeV" ) and any(y in key for y in ['2022','2024','2025']) and hasHarvest
        return result
upgradeWFs['seedingDeepCore'] = UpgradeWorkflow_seedingDeepCore(
    steps = [
        'Reco',
        'RecoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'RecoGlobal',
        'HARVESTGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'Nano',
        'ALCA',
    ],
    PU = [
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'HARVESTGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
    ],
    suffix = '_seedingDeepCore',
    offset = 0.17,
)

#Workflow to enable displacedRegionalStep tracking iteration
class UpgradeWorkflow_displacedRegional(UpgradeWorkflowTracking):
    def setup__(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return any(y in key for y in ['2022','2023','2024','2025'])
upgradeWFs['displacedRegional'] = UpgradeWorkflow_displacedRegional(
    steps = [
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
    ],
    PU = [],
    suffix = '_displacedRegional',
    offset = 0.701,
)
upgradeWFs['displacedRegional'].step3 = {
    '--procModifiers': 'displacedRegionalTracking'
}

# Vector Hits workflows
class UpgradeWorkflow_vectorHits(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = merge([{'--procModifiers': 'vectorHits'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="TTbar_14TeV" or fragment=="SingleMuPt10Extended") and 'Run4' in key
upgradeWFs['vectorHits'] = UpgradeWorkflow_vectorHits(
    steps = [
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    PU = [
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    suffix = '_vectorHits',
    offset = 0.9,
)

# WeightedMeanFitter vertexing workflows
class UpgradeWorkflow_weightedVertex(UpgradeWorkflow):
    def __init__(self, **kwargs):
        # adapt the parameters for the UpgradeWorkflow init method
        super(UpgradeWorkflow_weightedVertex, self).__init__(
            steps = [
                'Reco',
                'RecoFakeHLT',
                'HARVEST',
                'HARVESTFakeHLT',
                'RecoGlobal',
                'HARVESTGlobal',
                'RecoNano',
                'RecoNanoFakeHLT',
                'HARVESTNano',
                'HARVESTNanoFakeHLT',
            ],
            PU = [
                'Reco',
                'RecoFakeHLT',
                'HARVEST',
                'HARVESTFakeHLT',
                'RecoGlobal',
                'HARVESTGlobal',
                'RecoNano',
                'RecoNanoFakeHLT',
                'HARVESTNano',
                'HARVESTNanoFakeHLT',
            ],
            **kwargs)

    def setup_(self, step, stepName, stepDict, k, properties):
        # temporarily remove trigger & downstream steps
        if 'Reco' in step:
            mod = {'--procModifiers': 'weightedVertexing,vertexInBlocks', '--datatier':'GEN-SIM-RECO,DQMIO',
            '--eventcontent':'RECOSIM,DQM'}
            stepDict[stepName][k] = merge([mod,self.step3, stepDict[step][k]])
        if 'HARVEST' in step:
            stepDict[stepName][k] = merge([self.step4,stepDict[step][k]])

    def condition(self, fragment, stepList, key, hasHarvest):
        # select only a subset of the workflows
        selected = (fragment == "TTbar_14TeV") and ('FS' not in key) and hasHarvest
        result =  selected and any(y in key for y in ['2022','2024','2025','Run4'])

        return result


upgradeWFs['weightedVertex'] = UpgradeWorkflow_weightedVertex(
    suffix = '_weightedVertex',
    offset = 0.278,
)

upgradeWFs['weightedVertex'].step3 = {}
upgradeWFs['weightedVertex'].step4 = {}

upgradeWFs['weightedVertexTrackingOnly'] = UpgradeWorkflow_weightedVertex(
    suffix = '_weightedVertexTrackingOnly',
    offset = 0.279,
)

upgradeWFs['weightedVertexTrackingOnly'].step3 = {
    '-s': 'RAW2DIGI,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM',
    '--datatier':'GEN-SIM-RECO,DQMIO',
    '--eventcontent':'RECOSIM,DQM',
}

upgradeWFs['weightedVertexTrackingOnly'].step4 = {
    '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'
}

# Special TICL Pattern recognition Workflows
class UpgradeWorkflow_ticl_clue3D(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'RecoGlobal' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        if 'HARVESTGlobal' in step:
            stepDict[stepName][k] = merge([self.step4, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="TTbar_14TeV" or 'CloseByPGun_CE' in fragment) and 'Run4' in key
upgradeWFs['ticl_clue3D'] = UpgradeWorkflow_ticl_clue3D(
    steps = [
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    PU = [
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    suffix = '_ticl_clue3D',
    offset = 0.201,
)
upgradeWFs['ticl_clue3D'].step3 = {'--procModifiers': 'clue3D'}
upgradeWFs['ticl_clue3D'].step4 = {'--procModifiers': 'clue3D'}

class UpgradeWorkflow_ticl_FastJet(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'RecoGlobal' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        if 'HARVESTGlobal' in step:
            stepDict[stepName][k] = merge([self.step4, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="TTbar_14TeV" or 'CloseByPGun_CE' in fragment) and 'Run4' in key
upgradeWFs['ticl_FastJet'] = UpgradeWorkflow_ticl_FastJet(
    steps = [
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    PU = [
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    suffix = '_ticl_FastJet',
    offset = 0.202,
)
upgradeWFs['ticl_FastJet'].step3 = {'--procModifiers': 'fastJetTICL'}
upgradeWFs['ticl_FastJet'].step4 = {'--procModifiers': 'fastJetTICL'}

class UpgradeWorkflow_ticl_v5(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if ('Digi' in step and 'NoHLT' not in step) or ('HLTOnly' in step):
            stepDict[stepName][k] = merge([self.step2, stepDict[step][k]])
        if 'RecoGlobal' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        if 'HARVESTGlobal' in step:
            stepDict[stepName][k] = merge([self.step4, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="TTbar_14TeV" or 'CloseByP' in fragment or 'Eta1p7_2p7' in fragment) and 'Run4' in key

upgradeWFs['ticl_v5'] = UpgradeWorkflow_ticl_v5(
    steps = [
        'HLTOnly',
        'DigiTrigger',
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    PU = [
        'HLTOnly',
        'DigiTrigger',
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    suffix = '_ticl_v5',
    offset = 0.203,
)
upgradeWFs['ticl_v5'].step2 = {'--procModifiers': 'ticl_v5'}
upgradeWFs['ticl_v5'].step3 = {'--procModifiers': 'ticl_v5'}
upgradeWFs['ticl_v5'].step4 = {'--procModifiers': 'ticl_v5'}

class UpgradeWorkflow_ticl_v5_superclustering(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if ('Digi' in step and 'NoHLT' not in step) or ('HLTOnly' in step):
            stepDict[stepName][k] = merge([self.step2, stepDict[step][k]])
        if 'RecoGlobal' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        if 'HARVESTGlobal' in step:
            stepDict[stepName][k] = merge([self.step4, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="ZEE_14" or 'Eta1p7_2p7' in fragment) and 'Run4' in key
upgradeWFs['ticl_v5_superclustering_mustache_ticl'] = UpgradeWorkflow_ticl_v5_superclustering(
    steps = [
        'HLTOnly',
        'DigiTrigger',
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    PU = [
        'HLTOnly',
        'DigiTrigger',
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    suffix = '_ticl_v5_mustache',
    offset = 0.204,
)
upgradeWFs['ticl_v5_superclustering_mustache_ticl'].step2 = {'--procModifiers': 'ticl_v5,ticl_superclustering_mustache_ticl'}
upgradeWFs['ticl_v5_superclustering_mustache_ticl'].step3 = {'--procModifiers': 'ticl_v5,ticl_superclustering_mustache_ticl'}
upgradeWFs['ticl_v5_superclustering_mustache_ticl'].step4 = {'--procModifiers': 'ticl_v5,ticl_superclustering_mustache_ticl'}

upgradeWFs['ticl_v5_superclustering_mustache_pf'] = UpgradeWorkflow_ticl_v5_superclustering(
    steps = [
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    PU = [
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    suffix = '_ticl_v5_mustache_pf',
    offset = 0.205,
)
upgradeWFs['ticl_v5_superclustering_mustache_pf'].step3 = {'--procModifiers': 'ticl_v5,ticl_superclustering_mustache_pf'}
upgradeWFs['ticl_v5_superclustering_mustache_pf'].step4 = {'--procModifiers': 'ticl_v5,ticl_superclustering_mustache_pf'}

# Improved L2 seeding from L1Tk Muons and L3 Tracker Muon Inside-Out reconstruction first (Phase-2 Muon default)
class UpgradeWorkflow_phase2L2AndL3Muons(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if ('Digi' in step and 'NoHLT' not in step) or ('HLTOnly' in step):
            stepDict[stepName][k] = merge([self.step2, stepDict[step][k]])
        if 'RecoGlobal' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        if 'HARVESTGlobal' in step:
            stepDict[stepName][k] = merge([self.step4, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="ZMM_14" or 'SingleMu' in fragment or 'TTbar_14' in fragment) and 'Run4' in key

upgradeWFs['phase2L2AndL3Muons'] = UpgradeWorkflow_phase2L2AndL3Muons(
    steps = [
        'HLTOnly',
        'DigiTrigger',
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    PU = [
        'HLTOnly',
        'DigiTrigger',
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    suffix = '_phase2L2AndL3MuonsIOFirst',
    offset = 0.777,
)
upgradeWFs['phase2L2AndL3Muons'].step2 = {'--procModifiers':'phase2L2AndL3Muons'}
upgradeWFs['phase2L2AndL3Muons'].step3 = {'--procModifiers':'phase2L2AndL3Muons'}
upgradeWFs['phase2L2AndL3Muons'].step4 = {'--procModifiers':'phase2L2AndL3Muons'}

# Improved L2 seeding from L1Tk Muons and L3 Tracker Muon Outside-In reconstruction first
class UpgradeWorkflow_phase2L3MuonsOIFirst(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if ('Digi' in step and 'NoHLT' not in step) or ('HLTOnly' in step):
            stepDict[stepName][k] = merge([self.step2, stepDict[step][k]])
        if 'RecoGlobal' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        if 'HARVESTGlobal' in step:
            stepDict[stepName][k] = merge([self.step4, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="ZMM_14" or 'SingleMu' in fragment or 'TTbar_14' in fragment) and 'Run4' in key

upgradeWFs['phase2L3MuonsOIFirst'] = UpgradeWorkflow_phase2L3MuonsOIFirst(
    steps = [
        'HLTOnly',
        'DigiTrigger',
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    PU = [
        'HLTOnly',
        'DigiTrigger',
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    suffix = '_phase2L2AndL3MuonsOIFirst',
    offset = 0.778,
)
upgradeWFs['phase2L3MuonsOIFirst'].step2 = {'--procModifiers':'phase2L2AndL3Muons,phase2L3MuonsOIFirst'}
upgradeWFs['phase2L3MuonsOIFirst'].step3 = {'--procModifiers':'phase2L2AndL3Muons,phase2L3MuonsOIFirst'}
upgradeWFs['phase2L3MuonsOIFirst'].step4 = {'--procModifiers':'phase2L2AndL3Muons,phase2L3MuonsOIFirst'}

# Track DNN workflows
class UpgradeWorkflow_trackdnn(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = merge([{'--procModifiers': 'trackdnn'}, stepDict[step][k]])

    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and '2022' in key
upgradeWFs['trackdnn'] = UpgradeWorkflow_trackdnn(
    steps = [
        'Reco',
        'RecoFakeHLT',
        'RecoNano',
        'RecoNanoFakeHLT',
    ],
    PU = [
        'Reco',
        'RecoFakeHLT',
        'RecoNano',
        'RecoNanoFakeHLT',
    ],
    suffix = '_trackdnn',
    offset = 0.91,
)


# MLPF workflows
class UpgradeWorkflow_mlpf(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="TTbar_14TeV" or fragment=="QCD_FlatPt_15_3000HS_14") and '2022PU' in key

upgradeWFs['mlpf'] = UpgradeWorkflow_mlpf(
    steps = [
        'Reco',
        'RecoFakeHLT',
        'RecoNano',
        'RecoNanoFakeHLT',
    ],
    PU = [
        'Reco',
        'RecoFakeHLT',
        'RecoNano',
        'RecoNanoFakeHLT',
    ],
    suffix = '_mlpf',
    offset = 0.13,
)
upgradeWFs['mlpf'].step3 = {
    '--datatier': 'GEN-SIM-RECO,RECOSIM,MINIAODSIM,NANOAODSIM,DQMIO',
    '--eventcontent': 'FEVTDEBUGHLT,RECOSIM,MINIAODSIM,NANOEDMAODSIM,DQM',
    '--procModifiers': 'mlpf'
}


# ECAL DeepSC clustering studies workflow
class UpgradeWorkflow_ecalclustering(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="ZEE_14" or fragment=="TTbar_14TeV" or fragment=="WprimeTolNu_M3000_13TeV_pythia8"
            or fragment=="DisplacedSUSY_stopToBottom_M_300_1000mm_13" or fragment=="RunEGamma2018D" )

upgradeWFs['ecalDeepSC'] = UpgradeWorkflow_ecalclustering(
    steps = [
        'Reco',
        'RecoFakeHLT',
        'RecoNano',
        'RecoNanoFakeHLT',
    ],
    PU = [
        'Reco',
        'RecoFakeHLT',
        'RecoNano',
        'RecoNanoFakeHLT',
    ],
    suffix = '_ecalDeepSC',
    offset = 0.19,
)
upgradeWFs['ecalDeepSC'].step3 = {
    '--datatier': 'RECOSIM,MINIAODSIM,NANOAODSIM,DQMIO',
    '--eventcontent': 'RECOSIM,MINIAODSIM,NANOEDMAODSIM,DQM',
    '--procModifiers': 'ecal_deepsc'
}


# photonDRN workflows
class UpgradeWorkflow_photonDRN(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return '2018' in key and "SingleGamma" in fragment

upgradeWFs['photonDRN'] = UpgradeWorkflow_photonDRN(
    steps = [
        'RecoFakeHLT',
        'RecoNanoFakeHLT',
    ],
    PU = [
        'RecoFakeHLT',
        'RecoNanoFakeHLT',
    ],
    suffix = '_photonDRN',
    offset = 0.31,
)
upgradeWFs['photonDRN'].step3 = {
    '--procModifiers': 'enableSonicTriton,photonDRN'
}


# Patatrack workflows (NoPU and PU):
#   - TTbar_14, ZMM_14", ZEE_14, ZTT_14, NuGun, SingleMu, QCD_Pt15To7000_Flat for
#       > 2022, 2023, 2024, 2025 and Run4 conditions, TTbar
#   - Hydjet for HI conditions
class PatatrackWorkflow(UpgradeWorkflow):
    def __init__(self, digi = {}, reco = {}, mini = {}, harvest = {}, **kwargs):
        # adapt the parameters for the UpgradeWorkflow init method
        super(PatatrackWorkflow, self).__init__(
            steps = [
                'Digi',
                'HLTOnly',
                'DigiTrigger',
                'Reco',
                'RecoFakeHLT',
                'HARVEST',
                'HARVESTFakeHLT',
                'RecoGlobal',
                'HARVESTGlobal',
                'RecoNano',
                'RecoNanoFakeHLT',
                'HARVESTNano',
                'HARVESTNanoFakeHLT',
                'MiniAOD',
                'Nano',
                'ALCA',
                'ALCAPhase2'
            ],
            PU = [
                'Digi',
                'HLTOnly',
                'DigiTrigger',
                'Reco',
                'RecoFakeHLT',
                'HARVEST',
                'HARVESTFakeHLT',
                'RecoGlobal',
                'HARVESTGlobal',
                'RecoNano',
                'RecoNanoFakeHLT',
                'HARVESTNano',
                'HARVESTNanoFakeHLT',
                'MiniAOD',
                'Nano',
                'ALCA',
                'ALCAPhase2'
            ],
            **kwargs)
        self.__digi = digi
        self.__reco = reco
        if 'DQM' in self.__reco:
            self.__reco.update({
                '--datatier':     'GEN-SIM-RECO,DQMIO',
                '--eventcontent': 'RECOSIM,DQM'
            })
        self.__mini = mini
        self.__harvest = harvest

    def condition(self, fragment, stepList, key, hasHarvest):
        # select only a subset of the workflows
        years = ['2022','2023','2024','2025','Run4']
        fragments = ["TTbar_14","ZMM_14","ZEE_14","ZTT_14","NuGun","SingleMu","QCD_Pt15To7000_Flat"]
        selected = [
            (any(y in key for y in years) and ('FS' not in key) and any( f in fragment for f in fragments)),
            (('HI' in key) and ('Hydjet' in fragment) and ("PixelOnly" in self.suffix) )
        ]
        result = any(selected) and hasHarvest

        return result

    def setup_(self, step, stepName, stepDict, k, properties):
        # skip ALCA and Nano steps (but not RecoNano or HARVESTNano for Run3)
        if 'ALCA' in step or 'Nano'==step:
            stepDict[stepName][k] = None
        elif ('Digi' in step and "NoHLT" not in step) or 'HLTOnly' in step:
            if self.__digi is None:
              stepDict[stepName][k] = None
            else:
              stepDict[stepName][k] = merge([self.__digi, stepDict[step][k]])
        elif 'Reco' in step:
            if self.__reco is None:
              stepDict[stepName][k] = None
            else:
              stepDict[stepName][k] = merge([self.__reco, stepDict[step][k]])
            if 'Phase2' in stepDict[stepName][k]['--era']:
                if 'DQM:@standardDQM+@ExtraHLT' in stepDict[stepName][k]['-s']:
                    stepDict[stepName][k]['-s'] = stepDict[stepName][k]['-s'].replace('DQM:@standardDQM+@ExtraHLT','DQM:@phase2')
                if 'VALIDATION:@standardValidation' in stepDict[stepName][k]['-s']:
                    stepDict[stepName][k]['-s'] = stepDict[stepName][k]['-s'].replace('VALIDATION:@standardValidation','VALIDATION:@phase2Validation')


        elif 'MiniAOD' in step:
            if self.__mini is None:
              stepDict[stepName][k] = None
            else:
              stepDict[stepName][k] = merge([self.__mini, stepDict[step][k]])
        elif 'HARVEST' in step:
            if self.__harvest is None:
              stepDict[stepName][k] = None
            else:
              stepDict[stepName][k] = merge([self.__harvest, stepDict[step][k]])

###############################################################################################################
### Calorimeter-only reco
### these are not technically Patarack workflows but for
### the moment we can still leverage on the PatatrackWorkflow
### constructor for simplicity

# ECAL-only workflow running on CPU
#  - HLT on CPU
#  - ECAL-only reconstruction on CPU, with DQM and validation
#  - harvesting
upgradeWFs['ECALOnlyCPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_ecalOnly,RECO:reconstruction_ecalOnly,VALIDATION:@ecalOnlyValidation,DQM:@ecalOnly',
    },
    harvest = {
        '-s': 'HARVESTING:@ecalOnlyValidation+@ecal'
    },
    suffix = 'ECALOnlyCPU',
    offset = 0.511,
)

# HCAL-only workflow running on CPU
#  - HLT on CPU
#  - HCAL-only reconstruction on CPU, with DQM and validation
#  - harvesting
upgradeWFs['HCALOnlyCPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnly,VALIDATION:@hcalOnlyValidation,DQM:@hcalOnly+@hcal2Only',
    },
    harvest = {
        '-s': 'HARVESTING:@hcalOnlyValidation+@hcalOnly+@hcal2Only'
    },
    suffix = 'HCALOnlyCPU',
    offset = 0.521,
)

###############################################################################################################
### Alpaka workflows
###

# ECAL-only workflow running on CPU or GPU with Alpaka code
#  - HLT with Alpaka
#  - ECAL-only reconstruction with Alpaka, with DQM and validation
#  - harvesting
upgradeWFs['PatatrackECALOnlyAlpaka'] = PatatrackWorkflow(
    digi = {
        # customize the ECAL Local Reco part of the HLT menu for Alpaka
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_ecalOnly,RECO:reconstruction_ecalOnly,VALIDATION:@ecalOnlyValidation,DQM:@ecalOnly',
        '--procModifiers': 'alpaka',
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    harvest = {
        '-s': 'HARVESTING:@ecalOnlyValidation+@ecal'
    },
    suffix = 'Patatrack_ECALOnlyAlpaka',
    offset = 0.412,
)

# ECAL-only workflow running on CPU or GPU with Alpaka code
#  - HLT with Alpaka
#  - ECAL-only reconstruction with Alpaka on both CPU and GPU, with DQM and validation for GPU-vs-CPU comparisons
#  - harvesting
upgradeWFs['PatatrackECALOnlyAlpakaValidation'] = PatatrackWorkflow(
    digi = {
        # customize the ECAL Local Reco part of the HLT menu for Alpaka
        '--procModifiers': 'alpaka',
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_ecalOnly,RECO:reconstruction_ecalOnly,VALIDATION:@ecalOnlyValidation,DQM:@ecalOnly',
        '--procModifiers': 'alpakaValidation',
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    harvest = {
        '-s': 'HARVESTING:@ecalOnlyValidation+@ecal'
    },
    suffix = 'Patatrack_ECALOnlyAlpakaValidation',
    offset = 0.413,
)

# HCAL-PF Only workflow running HCAL local reco on GPU and PF with Alpaka with DQM and Validation
# - HLT-alpaka
# - HCAL-only reconstruction using Alpaka with DQM and Validation
upgradeWFs['PatatrackHCALOnlyAlpakaValidation'] = PatatrackWorkflow(
    digi = { 
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnly,VALIDATION:@hcalOnlyValidation,DQM:@hcalOnly+@hcal2Only',
        '--procModifiers': 'alpaka',
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    harvest = {
        '-s': 'HARVESTING:@hcalOnlyValidation'
    },
    suffix = 'Patatrack_HCALOnlyAlpaka_Validation',
    offset = 0.422,
)

# HCAL-PF Only workflow running HCAL local reco and PF with Alpaka with cluster level-validation
# - HLT-alpaka
# - HCAL-only reconstruction using GPU and Alpaka with DQM and Validation for PF Alpaka vs CPU comparisons
upgradeWFs['PatatrackHCALOnlyGPUandAlpakaValidation'] = PatatrackWorkflow(
    digi = {
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnlyLegacy+reconstruction_hcalOnly,VALIDATION:@hcalOnlyValidation+pfClusterHBHEOnlyAlpakaComparisonSequence,DQM:@hcalOnly+@hcal2Only+hcalOnlyOfflineSourceSequenceAlpaka',
        '--procModifiers': 'alpaka',
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    harvest = {
        '-s': 'HARVESTING:@hcalOnlyValidation'
    },
    suffix = 'Patatrack_HCALOnlyGPUandAlpaka_Validation',
    offset = 0.423,
)

# HCAL-PF Only workflow running HCAL local reco on CPU and PF with Alpaka slimmed for benchmarking
# - HLT-alpaka
# - HCAL-only reconstruction using Alpaka
upgradeWFs['PatatrackHCALOnlyAlpakaProfiling'] = PatatrackWorkflow(
    digi = {
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnly',
        '--procModifiers': 'alpaka'
    },
    harvest = None,
    suffix = 'Patatrack_HCALOnlyAlpaka_Profiling',
    offset = 0.424,
)

# Workflow running the Pixel quadruplets, ECAL and HCAL reconstruction on GPU (optional), PF using Alpaka, together with the full offline reconstruction on CPU
#  - HLT on GPU (optional)
#  - reconstruction on Alpaka, with DQM and validation
#  - harvesting
upgradeWFs['PatatrackFullRecoAlpaka'] = PatatrackWorkflow(
    digi = {
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    reco = {
        # skip the @pixelTrackingOnlyValidation which cannot run together with the full reconstruction
        '-s': 'RAW2DIGI:RawToDigi+RawToDigi_pixelOnly,L1Reco,RECO:reconstruction+reconstruction_pixelTrackingOnly,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@pixelTrackingOnlyDQM',
        '--procModifiers': 'alpaka',
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    harvest = {
        # skip the @pixelTrackingOnlyDQM harvesting
    },
    suffix = 'Patatrack_FullRecoAlpaka',
    offset = 0.492,
)

# Workflow running the Pixel triplets, ECAL and HCAL reconstruction on GPU (optional), PF using Alpaka, together with the full offline reconstruction on CPU
#  - HLT on GPU (optional)
#  - reconstruction on Alpaka, with DQM and validation
#  - harvesting

upgradeWFs['PatatrackFullRecoAlpaka'] = PatatrackWorkflow(
    digi = {
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    reco = {
        # skip the @pixelTrackingOnlyValidation which cannot run together with the full reconstruction
        '-s': 'RAW2DIGI:RawToDigi+RawToDigi_pixelOnly,L1Reco,RECO:reconstruction+reconstruction_pixelTrackingOnly,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@pixelTrackingOnlyDQM',
        '--procModifiers': 'alpaka',
        '--customise' : 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets,HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    harvest = {
        # skip the @pixelTrackingOnlyDQM harvesting
    },
    suffix = 'Patatrack_FullRecoAlpakaTriplets',
    offset = 0.496,
)

# Pixel-only quadruplets workflow running on GPU (optional)
#  - Pixel-only reconstruction with Alpaka, with DQM and validation
#  - harvesting

upgradeWFs['PatatrackPixelOnlyAlpaka'] = PatatrackWorkflow(
    digi = {
        '--procModifiers': 'alpaka', 
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
        '--procModifiers': 'alpaka',
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'
    },
    suffix = 'Patatrack_PixelOnlyAlpaka',
    offset = 0.402,
)

# Pixel-only quadruplets workflow running on GPU (optional)
#  - Pixel-only reconstruction with Alpaka, with standard and CPUvsGPU DQM and validation
#  - harvesting for CPUvsGPU validation

upgradeWFs['PatatrackPixelOnlyAlpakaValidation'] = PatatrackWorkflow(
    digi = {
        '--procModifiers': 'alpaka', 
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
        '--procModifiers': 'alpakaValidation',
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM',
        '--procModifiers': 'alpakaValidation',
    },
    suffix = 'Patatrack_PixelOnlyAlpaka_Validation',
    offset = 0.403,
)

# Pixel-only quadruplets workflow running on CPU or GPU, trimmed down for benchmarking

upgradeWFs['PatatrackPixelOnlyAlpakaProfiling'] = PatatrackWorkflow(
    digi = {
        '--procModifiers': 'alpaka', 
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly',
        '--procModifiers': 'alpaka',
        '--customise' : 'RecoTracker/Configuration/customizePixelOnlyForProfiling.customizePixelOnlyForProfilingGPUOnly'
    },
    harvest = None,
    suffix = 'Patatrack_PixelOnlyAlpaka_Profiling',
    offset = 0.404,
)


# Pixel-only triplets workflow running on GPU (optional)
#  - Pixel-only reconstruction with Alpaka, with standard and CPUvsGPU DQM and validation
#  - harvesting for CPUvsGPU validation

upgradeWFs['PatatrackPixelOnlyTripletsAlpaka'] = PatatrackWorkflow(
    digi = {
        '--procModifiers': 'alpaka',
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
        '--procModifiers': 'alpaka',
        '--customise' : 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets,HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'
    },
    suffix = 'Patatrack_PixelOnlyTripletsAlpaka',
    offset = 0.406,
)

# Pixel-only triplets workflow running on GPU (optional)
#  - Pixel-only reconstruction with Alpaka, with standard and CPUvsGPU DQM and validation
#  - harvesting for CPUvsGPU validation

upgradeWFs['PatatrackPixelOnlyTripletsAlpakaValidation'] = PatatrackWorkflow(
    digi = { 
        '--procModifiers': 'alpaka',
        '--customise' : 'HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling',
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
        '--procModifiers': 'alpakaValidation',
        '--customise' : 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets,HeterogeneousCore/AlpakaServices/customiseAlpakaServiceMemoryFilling.customiseAlpakaServiceMemoryFilling'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'
    },
    suffix = 'Patatrack_PixelOnlyTripletsAlpaka_Validation',
    offset = 0.407,
)

upgradeWFs['PatatrackPixelOnlyTripletsAlpakaProfiling'] = PatatrackWorkflow(
    digi = { 
        '--procModifiers': 'alpaka',
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly',
        '--procModifiers': 'alpaka',
        '--customise' : 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets,RecoTracker/Configuration/customizePixelOnlyForProfiling.customizePixelOnlyForProfilingGPUOnly'
    },
    harvest = None,
    suffix = 'Patatrack_PixelOnlyTripletsAlpaka_Profiling',
    offset = 0.408,
)

# end of Patatrack workflows
###############################################################################################################

class UpgradeWorkflow_ProdLike(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        thisStep = stepDict[step][k]["-s"]
        if 'GenSimHLBeamSpot14' in step:
            stepDict[stepName][k] = merge([{'--eventcontent': 'RAWSIM', '--datatier': 'GEN-SIM'},stepDict[step][k]])
        elif 'Digi' in step and 'Trigger' not in step:
            stepDict[stepName][k] = merge([{'-s': thisStep.replace("DIGI:pdigi_valid","DIGI"),'--datatier':'GEN-SIM-RAW', '--eventcontent':'RAWSIM'}, stepDict[step][k]])
        elif 'DigiTrigger' in step: # for Phase-2
            stepDict[stepName][k] = merge([{'-s': thisStep.replace("DIGI:pdigi_valid","DIGI"), '--datatier':'GEN-SIM-RAW', '--eventcontent':'RAWSIM'}, stepDict[step][k]])
        elif 'Reco' in step:
            stepDict[stepName][k] = merge([{'-s': 'RAW2DIGI,L1Reco,RECO,RECOSIM', '--datatier':'AODSIM', '--eventcontent':'AODSIM'}, stepDict[step][k]])
        elif 'MiniAOD' in step:
            # the separate miniAOD step is used here
            stepDict[stepName][k] = deepcopy(stepDict[step][k])
        elif 'ALCA' in step or 'HARVEST' in step:
            # remove step
            stepDict[stepName][k] = None
        elif 'Nano'==step:
            stepDict[stepName][k] = merge([{'--filein':'file:step4.root','-s':'NANO','--datatier':'NANOAODSIM','--eventcontent':'NANOEDMAODSIM'}, stepDict[step][k]])
    def setupPU_(self, step, stepName, stepDict, k, properties):
        # No need for PU replay for ProdLike
        if "Digi" not in step and stepDict[stepName][k] is not None and '--pileup' in stepDict[stepName][k]:
            stepDict[stepName][k].pop('--pileup', None)
            stepDict[stepName][k].pop('--pileup_input', None)
    def condition(self, fragment, stepList, key, hasHarvest):
        years = ['2022','2023','2024','2025','Run4']
        return fragment=="TTbar_14TeV" and any(y in key for y in years)
upgradeWFs['ProdLike'] = UpgradeWorkflow_ProdLike(
    steps = [
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'MiniAOD',
        'ALCA',
        'ALCAPhase2',
        'Nano',
    ],
    PU = [
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'MiniAOD',
        'ALCA',
        'ALCAPhase2',
        'Nano',
    ],
    suffix = '_ProdLike',
    offset = 0.21,
)

class UpgradeWorkflow_ProdLikeRunningPU(UpgradeWorkflow_ProdLike):
    def __init__(self, suffix, offset, fixedPU,
         steps = [],
        PU = [
            'GenSimHLBeamSpot14',
            'Digi',
            'DigiTrigger',
            'Reco',
            'RecoFakeHLT',
            'RecoGlobal',
            'RecoNano',
            'RecoNanoFakeHLT',
            'HARVEST',
            'HARVESTFakeHLT',
            'HARVESTGlobal',
            'HARVESTNano',
            'HARVESTNanoFakeHLT',
            'MiniAOD',
            'ALCA',
            'ALCAPhase2',
            'Nano',
        ]):
        super(UpgradeWorkflow_ProdLikeRunningPU, self).__init__(steps, PU, suffix, offset)
        self.__fixedPU = fixedPU
    def setupPU_(self, step, stepName, stepDict, k, properties):
        #  change PU skipping ALCA and HARVEST
        if stepDict[stepName][k] is not None and '--pileup' in stepDict[stepName][k] and "Digi" in step:
            stepDict[stepName][k]['--pileup'] = 'AVE_' + str(self.__fixedPU) + '_BX_25ns'
    def condition(self, fragment, stepList, key, hasHarvest):
        # lower PUs for Run3
        return (fragment=="TTbar_14TeV") and (('Run4' in key) or ('2022' in key and self.__fixedPU<=100))

# The numbering below is following the 0.21 for ProdLike wfs
# 0.21N would have been a more natural choice but the
# trailing zeros are ignored. Thus 0.21N1 is used

upgradeWFs['ProdLikePU10'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU10',
    offset = 0.21101,
    fixedPU = 10,
)

upgradeWFs['ProdLikePU20'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU20',
    offset = 0.21201,
    fixedPU = 20,
)

upgradeWFs['ProdLikePU30'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU30',
    offset = 0.21301,
    fixedPU = 30,
)

upgradeWFs['ProdLikePU40'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU40',
    offset = 0.21401,
    fixedPU = 40,
)

upgradeWFs['ProdLikePU50'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU50',
    offset = 0.21501,
    fixedPU = 50,
)

upgradeWFs['ProdLikePU55'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU55',
    offset = 0.21551,
    fixedPU = 55,
)

upgradeWFs['ProdLikePU60'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU60',
    offset = 0.21601,
    fixedPU = 60,
)

upgradeWFs['ProdLikePU65'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU65',
    offset = 0.21651,
    fixedPU = 65,
)

upgradeWFs['ProdLikePU70'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU70',
    offset = 0.21701,
    fixedPU = 70,
)

upgradeWFs['ProdLikePU80'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU80',
    offset = 0.21801,
    fixedPU = 80,
)

upgradeWFs['ProdLikePU90'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU90',
    offset = 0.21901,
    fixedPU = 90,
)

upgradeWFs['ProdLikePU100'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU100',
    offset = 0.211001,
    fixedPU = 100,
)

upgradeWFs['ProdLikePU120'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU120',
    offset = 0.211201,
    fixedPU = 120,
)

upgradeWFs['ProdLikePU140'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU140',
    offset = 0.211401,
    fixedPU = 140,
)

upgradeWFs['ProdLikePU160'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU160',
    offset = 0.211601,
    fixedPU = 160,
)

upgradeWFs['ProdLikePU180'] = UpgradeWorkflow_ProdLikeRunningPU(
    suffix = '_ProdLikePU180',
    offset = 0.211801,
    fixedPU = 180,
)

class UpgradeWorkflow_HLT75e33Timing(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        # skip RECO, ALCA and HARVEST
        if ('ALCA' in step) or ('Reco' in step) or ('HARVEST' in step) or ('HLT' in step):
            stepDict[stepName][k] = None
        elif 'DigiTrigger' in step:
            stepDict[stepName][k] = merge([self.step2, stepDict[step][k]])
        else:
            stepDict[stepName][k] = merge([stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and 'Run4' in key
upgradeWFs['HLTTiming75e33'] = UpgradeWorkflow_HLT75e33Timing(
    steps = [
        'Reco',
        'RecoGlobal',
        'RecoNano',
        'DigiTrigger',
        'ALCA',
        'ALCAPhase2',
        'HARVESTGlobal',
    ],
    PU = [
        'Reco',
        'RecoGlobal',
        'RecoNano',
        'DigiTrigger',
        'ALCA',
        'ALCAPhase2',
        'HARVESTGlobal'
    ],
    suffix = '_HLT75e33Timing',
    offset = 0.75,
)
upgradeWFs['HLTTiming75e33'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing'
}

upgradeWFs['HLTTiming75e33Alpaka'] = deepcopy(upgradeWFs['HLTTiming75e33'])
upgradeWFs['HLTTiming75e33Alpaka'].suffix = '_HLT75e33TimingAlpaka'
upgradeWFs['HLTTiming75e33Alpaka'].offset = 0.751
upgradeWFs['HLTTiming75e33Alpaka'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing',
    '--procModifiers': 'alpaka'
}

upgradeWFs['HLTTiming75e33TiclV5'] = deepcopy(upgradeWFs['HLTTiming75e33'])
upgradeWFs['HLTTiming75e33TiclV5'].suffix = '_HLT75e33TimingTiclV5'
upgradeWFs['HLTTiming75e33TiclV5'].offset = 0.752
upgradeWFs['HLTTiming75e33TiclV5'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing',
    '--procModifiers': 'ticl_v5'
}

upgradeWFs['HLTTiming75e33AlpakaSingleIter'] = deepcopy(upgradeWFs['HLTTiming75e33'])
upgradeWFs['HLTTiming75e33AlpakaSingleIter'].suffix = '_HLT75e33TimingAlpakaSingleIter'
upgradeWFs['HLTTiming75e33AlpakaSingleIter'].offset = 0.753
upgradeWFs['HLTTiming75e33AlpakaSingleIter'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing',
    '--procModifiers': 'alpaka,singleIterPatatrack'
}

upgradeWFs['HLTTiming75e33AlpakaSingleIterLST'] = deepcopy(upgradeWFs['HLTTiming75e33'])
upgradeWFs['HLTTiming75e33AlpakaSingleIterLST'].suffix = '_HLT75e33TimingAlpakaSingleIterLST'
upgradeWFs['HLTTiming75e33AlpakaSingleIterLST'].offset = 0.754
upgradeWFs['HLTTiming75e33AlpakaSingleIterLST'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing',
    '--procModifiers': 'alpaka,singleIterPatatrack,trackingLST'
}

upgradeWFs['HLTTiming75e33AlpakaLST'] = deepcopy(upgradeWFs['HLTTiming75e33'])
upgradeWFs['HLTTiming75e33AlpakaLST'].suffix = '_HLT75e33TimingAlpakaLST'
upgradeWFs['HLTTiming75e33AlpakaLST'].offset = 0.755
upgradeWFs['HLTTiming75e33AlpakaLST'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing',
    '--procModifiers': 'alpaka,trackingLST'
}

upgradeWFs['HLTTiming75e33TrimmedTracking'] = deepcopy(upgradeWFs['HLTTiming75e33'])
upgradeWFs['HLTTiming75e33TrimmedTracking'].suffix = '_HLT75e33TimingTrimmedTracking'
upgradeWFs['HLTTiming75e33TrimmedTracking'].offset = 0.756
upgradeWFs['HLTTiming75e33TrimmedTracking'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing',
    '--procModifiers': 'phase2_hlt_vertexTrimming'
}

upgradeWFs['HLTTiming75e33AlpakaTrimmedTracking'] = deepcopy(upgradeWFs['HLTTiming75e33'])
upgradeWFs['HLTTiming75e33AlpakaTrimmedTracking'].suffix = '_HLT75e33TimingAlpakaTrimmedTracking'
upgradeWFs['HLTTiming75e33AlpakaTrimmedTracking'].offset = 0.7561
upgradeWFs['HLTTiming75e33AlpakaTrimmedTracking'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing',
    '--procModifiers': 'alpaka,phase2_hlt_vertexTrimming'
}

upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingSingleIter'] = deepcopy(upgradeWFs['HLTTiming75e33'])
upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingSingleIter'].suffix = '_HLT75e33TimingAlpakaTrimmedTrackingSingleIter'
upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingSingleIter'].offset = 0.7562
upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingSingleIter'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing',
    '--procModifiers': 'alpaka,phase2_hlt_vertexTrimming,singleIterPatatrack'
}

upgradeWFs['HLTTiming75e33TrimmedTrackingLST'] = deepcopy(upgradeWFs['HLTTiming75e33'])
upgradeWFs['HLTTiming75e33TrimmedTrackingLST'].suffix = '_HLT75e33TimingTrimmedTrackingLST'
upgradeWFs['HLTTiming75e33TrimmedTrackingLST'].offset = 0.7563
upgradeWFs['HLTTiming75e33TrimmedTrackingLST'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing',
    '--procModifiers': 'phase2_hlt_vertexTrimming,trackingLST'
}

upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingLST'] = deepcopy(upgradeWFs['HLTTiming75e33'])
upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingLST'].suffix = '_HLT75e33TimingAlpakaTrimmedTrackingLST'
upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingLST'].offset = 0.7564
upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingLST'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing',
    '--procModifiers': 'alpaka,phase2_hlt_vertexTrimming,trackingLST'
}

upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingSingleIterLST'] = deepcopy(upgradeWFs['HLTTiming75e33'])
upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingSingleIterLST'].suffix = '_HLT75e33TimingAlpakaTrimmedTrackingSingleIterLST'
upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingSingleIterLST'].offset = 0.7565
upgradeWFs['HLTTiming75e33AlpakaTrimmedTrackingSingleIterLST'].step2 = {
    '-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,L1P2GT,DIGI2RAW,HLT:75e33_timing',
    '--procModifiers': 'alpaka,phase2_hlt_vertexTrimming,singleIterPatatrack,trackingLST'
}


class UpgradeWorkflow_HLTwDIGI75e33(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'DigiTrigger' in step:
            stepDict[stepName][k] = merge([{'-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,DIGI2RAW,HLT:@relvalRun4'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and 'Run4' in key
upgradeWFs['HLTwDIGI75e33'] = UpgradeWorkflow_HLTwDIGI75e33(
    steps = [
        'DigiTrigger',
    ],
    PU = [
        'DigiTrigger',
    ],
    suffix = '_HLTwDIGI75e33',
    offset = 0.76,
)

class UpgradeWorkflow_L1Complete(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Digi' in step and 'NoHLT' not in step:
            stepDict[stepName][k] = merge([{'-s': 'DIGI:pdigi_valid,L1,L1TrackTrigger,L1P2GT,DIGI2RAW,HLT:@relvalRun4'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return 'Run4' in key

upgradeWFs['L1Complete'] = UpgradeWorkflow_L1Complete(
    steps = [
        'DigiTrigger',
    ],
    PU = [
        'DigiTrigger',
    ],
    suffix = '_L1Complete',
    offset = 0.78
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
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
    ],
    PU = [
        'Digi',
        'DigiTrigger',
    ],
    suffix = '_Neutron',
    offset = 0.12,
)
# add some extra info
upgradeWFs['Neutron'].neutronKeys = [x for x in upgradeKeys['Run4'] if 'PU' not in x]
upgradeWFs['Neutron'].neutronFrags = ['ZMM_14','MinBias_14TeV']

class UpgradeWorkflow_heCollapse(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = merge([{'--procModifiers': 'run2_HECollapse_2018'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_13" and '2018' in key
upgradeWFs['heCollapse'] = UpgradeWorkflow_heCollapse(
    steps = [
        'GenSim',
        'Digi',
        'Reco',
#        'RecoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'ALCA',
    ],
    PU = [
        'Digi',
        'Reco',
#        'RecoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
    ],
    suffix = '_heCollapse',
    offset = 0.6,
)

# ECAL Phase 2 development WF
class UpgradeWorkflow_ecalDevel(UpgradeWorkflow):
    def __init__(self, digi = {}, reco = {}, harvest = {}, **kwargs):
        # adapt the parameters for the UpgradeWorkflow init method
        super(UpgradeWorkflow_ecalDevel, self).__init__(
            steps = [
                'DigiTrigger',
                'RecoGlobal',
                'HARVESTGlobal',
                'ALCAPhase2',
            ],
            PU = [
                'DigiTrigger',
                'RecoGlobal',
                'HARVESTGlobal',
                'ALCAPhase2',
            ],
            **kwargs)
        self.__digi = digi
        self.__reco = reco
        self.__harvest = harvest

    def setup_(self, step, stepName, stepDict, k, properties):
        # temporarily remove trigger & downstream steps
        mods = {'--era': stepDict[step][k]['--era']+',phase2_ecal_devel'}
        if 'Digi' in step:
            mods['-s'] = 'DIGI:pdigi_valid,DIGI2RAW'
            mods |= self.__digi
        elif 'Reco' in step:
            mods['-s'] = 'RAW2DIGI,RECO:reconstruction_ecalOnly,VALIDATION:@ecalOnlyValidation,DQM:@ecalOnly'
            mods['--datatier'] = 'GEN-SIM-RECO,DQMIO'
            mods['--eventcontent'] = 'FEVTDEBUGHLT,DQM'
            mods |= self.__reco
        elif 'HARVEST' in step:
            mods['-s'] = 'HARVESTING:@ecalOnlyValidation+@ecal'
            mods |= self.__harvest
        stepDict[stepName][k] = merge([mods, stepDict[step][k]])
        # skip ALCA step
        if 'ALCA' in step:
            stepDict[stepName][k] = None

    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and 'Run4' in key

# ECAL Phase 2 workflow running on CPU
upgradeWFs['ecalDevel'] = UpgradeWorkflow_ecalDevel(
    suffix = '_ecalDevel',
    offset = 0.61,
)

# ECAL Phase 2 workflow running on CPU or GPU (if available)
upgradeWFs['ecalDevelGPU'] = UpgradeWorkflow_ecalDevel(
    reco = {'--procModifiers': 'gpu'},
    suffix = '_ecalDevelGPU',
    offset = 0.612,
)

# ECAL component
class UpgradeWorkflow_ECalComponent(UpgradeWorkflow):
    def __init__(self, suffix, offset, ecalTPPh2, ecalMod,
                 steps = [
                     'GenSim',
                     'GenSimHLBeamSpot',
                     'GenSimHLBeamSpot14',
                     'GenSimHLBeamSpotHGCALCloseBy',
                     'Digi',
                     'DigiTrigger',
                     'RecoGlobal',
                     'HARVESTGlobal',
                     'ALCAPhase2',
                 ],
                 PU = [
                     'GenSim',
                     'GenSimHLBeamSpot',
                     'GenSimHLBeamSpot14',
                     'GenSimHLBeamSpotHGCALCloseBy',
                     'Digi',
                     'DigiTrigger',
                     'RecoGlobal',
                     'HARVESTGlobal',
                     'ALCAPhase2',
                 ]):
        super(UpgradeWorkflow_ECalComponent, self).__init__(steps, PU, suffix, offset)
        self.__ecalTPPh2 = ecalTPPh2
        self.__ecalMod = ecalMod

    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = deepcopy(stepDict[step][k])
        if 'Sim' in step:
            if self.__ecalMod is not None:
                stepDict[stepName][k] = merge([{'--procModifiers':self.__ecalMod},stepDict[step][k]])
        if 'Digi' in step and 'NoHLT' not in step:
            if self.__ecalMod is not None:
                stepDict[stepName][k] = merge([{'--procModifiers':self.__ecalMod},stepDict[step][k]])
            if self.__ecalTPPh2 is not None:
                mods = {'--era': stepDict[step][k]['--era']+',phase2_ecal_devel,phase2_ecalTP_devel'}
                mods['-s'] = 'DIGI:pdigi_valid,DIGI2RAW,HLT:@fake2'
                stepDict[stepName][k] = merge([mods, stepDict[step][k]])
        if 'RecoGlobal' in step:
            stepDict[stepName][k] = merge([{'-s': 'RAW2DIGI,RECO,RECOSIM,PAT',
                                            '--datatier':'GEN-SIM-RECO',
                                            '--eventcontent':'FEVTDEBUGHLT',
                                        }, stepDict[step][k]])
        if 'HARVESTGlobal' in step:
            stepDict[stepName][k] = None
        if 'ALCAPhase2' in step:
            stepDict[stepName][k] = None

    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and ('2022' in key or '2023' in key or 'Run4' in key)

upgradeWFs['ECALComponent'] = UpgradeWorkflow_ECalComponent(
    suffix = '_ecalComponent',
    offset = 0.631,
    ecalTPPh2 = None,
    ecalMod = 'ecal_component',
)

upgradeWFs['ECALComponentFSW'] = UpgradeWorkflow_ECalComponent(
    suffix = '_ecalComponentFSW',
    offset = 0.632,
    ecalTPPh2 = None,
    ecalMod = 'ecal_component_finely_sampled_waveforms',
)

upgradeWFs['ECALTPPh2'] = UpgradeWorkflow_ECalComponent(
    suffix = '_ecalTPPh2',
    offset = 0.633,
    ecalTPPh2 = 'phase2_ecal_devel,phase2_ecalTP_devel',
    ecalMod = None,
)

upgradeWFs['ECALTPPh2Component'] = UpgradeWorkflow_ECalComponent(
    suffix = '_ecalTPPh2Component',
    offset = 0.634,
    ecalTPPh2 = 'phase2_ecal_devel,phase2_ecalTP_devel',
    ecalMod = 'ecal_component',
)

upgradeWFs['ECALTPPh2ComponentFSW'] = UpgradeWorkflow_ECalComponent(
    suffix = '_ecalTPPh2ComponentFSW',
    offset = 0.635,
    ecalTPPh2 = 'phase2_ecal_devel,phase2_ecalTP_devel',
    ecalMod = 'ecal_component_finely_sampled_waveforms',
)

class UpgradeWorkflow_0T(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        myGT=stepDict[step][k]['--conditions']
        myGT+="_0T"
        stepDict[stepName][k] = merge([{'-n':'1','--magField':'0T','--conditions':myGT}, stepDict[step][k]])
    def setupPU_(self, step, stepName, stepDict, k, properties):
        # override '-n' setting from PUDataSets in relval_steps.py
        stepDict[stepName][k] = merge([{'-n':'1'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="TTbar_13" or fragment=="TTbar_14TeV") and ('2017' in key or '2018' in key or '2022' in key or '2024' in key) and ('FS' not in key)
upgradeWFs['0T'] = UpgradeWorkflow_0T(
    steps = [
        'GenSim',
        'Digi',
        'Reco',
        'RecoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'ALCA',
    ],
    PU = [
        'Digi',
        'Reco',
        'RecoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
    ],
    suffix = '_0T',
    offset = 0.24,
)

class UpgradeWorkflow_ParkingBPH(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step and 'Run2_2018' in stepDict[step][k]['--era']:
            stepDict[stepName][k] = merge([{'--era': 'Run2_2018,bParking'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_13" and '2018' in key
upgradeWFs['ParkingBPH'] = UpgradeWorkflow_ParkingBPH(
    steps = [
        'Reco',
        'RecoFakeHLT',
    ],
    PU = [],
    suffix = '_ParkingBPH',
    offset = 0.8,
)

## Wf to add Heavy Flavor DQM to whichever DQM is already there
class UpgradeWorkflow_HeavyFlavor(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        self.__frags = ["B0","Psi2S","Bu","Bd","Xi","Bs"]
        thisStep = stepDict[step][k]["-s"]
        if "Reco" in step:
            if "DQM:" in thisStep:
                stepDict[stepName][k] = merge([{'-s': thisStep.replace("DQM:","DQM:@heavyFlavor+")}, stepDict[step][k]])
            elif "DQM" in thisStep:
                stepDict[stepName][k] = merge([{'-s': thisStep.replace("DQM","DQM:@heavyFlavor")}, stepDict[step][k]])
            else:
                stepDict[stepName][k] = merge([{'-s': thisStep + ",DQM:@heavyFlavor"}, stepDict[step][k]])

    def condition(self, fragment, stepList, key, hasHarvest):
        return any(frag in fragment for frag in self.__frags)

upgradeWFs['HeavyFlavor'] = UpgradeWorkflow_HeavyFlavor(
    steps = [
        'Reco',
        'RecoFakeHLT',
        'RecoNano',
        'RecoNanoFakeHLT',
    ],
    PU = [],
    suffix = '_HeavyFlavor',
    offset = 0.81,
)


class UpgradeWorkflow_JMENano(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Nano' in step:
            stepDict[stepName][k] = merge([{'--customise': 'PhysicsTools/NanoAOD/custom_jme_cff.PrepJMECustomNanoAOD'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="TTbar_13" or fragment=="TTbar_14TeV") and ('2017' in key or '2018' in key or '2022' in key) and ('FS' not in key)
upgradeWFs['JMENano'] = UpgradeWorkflow_JMENano(
    steps = [
        'Nano',
        'RecoNano',
        'RecoNanoFakeHLT',
    ],
    PU = [],
    suffix = '_JMENano',
    offset = 0.15,
)


# common operations for aging workflows
class UpgradeWorkflowAging(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Digi' in step or 'Reco' in step:
            stepDict[stepName][k] = merge([{'--customise': 'SLHCUpgradeSimulations/Configuration/aging.customise_aging_'+self.lumi}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return 'Run4' in key
# define several of them
upgradeWFs['Aging1000'] = UpgradeWorkflowAging(
    steps =  [
        'Digi',
        'DigiTrigger',
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
    ],
    PU =  [
        'Digi',
        'DigiTrigger',
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
    ],
    suffix = 'Aging1000',
    offset = 0.101,
)
upgradeWFs['Aging1000'].lumi = '1000'
upgradeWFs['Aging3000'] = deepcopy(upgradeWFs['Aging1000'])
upgradeWFs['Aging3000'].suffix = 'Aging3000'
upgradeWFs['Aging3000'].offset = 0.103
upgradeWFs['Aging3000'].lumi = '3000'

class UpgradeWorkflow_PixelClusterSplitting(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = merge([{'--procModifiers': 'splitClustersInPhase2Pixel'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return 'Run4' in key

upgradeWFs['PixelClusterSplitting'] = UpgradeWorkflow_PixelClusterSplitting(
    steps = [
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
    ],
    PU = [
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
    ],
    suffix = '_ClusterSplittingInPixel',
    offset = 0.19001,
)

class UpgradeWorkflow_JetCore(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = merge([{'--procModifiers': 'splitClustersInPhase2Pixel,jetCoreInPhase2'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return 'Run4' in key

upgradeWFs['JetCore'] = UpgradeWorkflow_JetCore(
    steps = [
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
    ],
    PU = [
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
    ],
    suffix = '_JetCore',
    offset = 0.19002,
)

class UpgradeWorkflow_SplittingFromHLT(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = merge([{'--procModifiers': 'hltClusterSplitting'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return '2025' in key and fragment=="TTbar_14TeV"

upgradeWFs['SplittingFromHLT'] = UpgradeWorkflow_SplittingFromHLT(
    steps = [
        'DigiTrigger',
        'Digi',
        'HLTOnly',
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
    ],
    PU = [
        'DigiTrigger',
        'Digi',
        'HLTOnly',
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
    ],
    suffix = '_SplittingFromHLT',
    offset = 0.19003,
)

class UpgradeWorkflow_SplittingProdLike(UpgradeWorkflow_ProdLike):
    def __init__(self, suffix, offset,steps, PU):
        super(UpgradeWorkflow_SplittingProdLike, self).__init__(steps, PU, suffix, offset)

    def setup_(self, step, stepName, stepDict, k, properties):
        # copy steps, then apply specializations
        stepDict[stepName][k] = merge([{'--procModifiers': 'hltClusterSplitting'}, stepDict[step][k]])

    def condition(self, fragment, stepList, key, hasHarvest):
        return '2025' in key and fragment=="TTbar_14TeV"

upgradeWFs['SplittingFromHLTProdLike'] = UpgradeWorkflow_SplittingProdLike(
    steps = [
    ],
    PU = [
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'HLTOnly',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'MiniAOD',
        'ALCA',
        'Nano',
    ],
    suffix = '_SplittingFromHLTProdLike',
    offset = 0.1900321,
)

#
# Simulates Bias Rail in Phase-2 OT PS modules and X% random bad Strips
# in PS-s and SS sensors
#
class UpgradeWorkflow_OTInefficiency(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Digi' in step:
            stepDict[stepName][k] = merge([{'--customise': 'SimTracker/SiPhase2Digitizer/customizeForOTInefficiency.customizeSiPhase2OTInefficiency'+self.percent+'Percent'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and 'Run4' in key
# define several of them
upgradeWFs['OTInefficiency'] = UpgradeWorkflow_OTInefficiency(
    steps =  [
        'Digi',
        'DigiTrigger',
    ],
    PU =  [
        'Digi',
        'DigiTrigger',
    ],
    suffix = '_OTInefficiency',
    offset = 0.111,
)
upgradeWFs['OTInefficiency'].percent = 'Zero'

# 1% bad strips
upgradeWFs['OTInefficiency1PC'] = deepcopy(upgradeWFs['OTInefficiency'])
upgradeWFs['OTInefficiency1PC'].suffix = '_OTInefficiency1PC'
upgradeWFs['OTInefficiency1PC'].offset = 0.112
upgradeWFs['OTInefficiency1PC'].percent = 'One'

# 5% bad strips
upgradeWFs['OTInefficiency5PC'] = deepcopy(upgradeWFs['OTInefficiency'])
upgradeWFs['OTInefficiency5PC'].suffix = '_OTInefficiency5PC'
upgradeWFs['OTInefficiency5PC'].offset = 0.113
upgradeWFs['OTInefficiency5PC'].percent = 'Five'

# 10% bad strips
upgradeWFs['OTInefficiency10PC'] = deepcopy(upgradeWFs['OTInefficiency'])
upgradeWFs['OTInefficiency10PC'].suffix = '_OTInefficiency10PC'
upgradeWFs['OTInefficiency10PC'].offset = 0.114
upgradeWFs['OTInefficiency10PC'].percent = 'Ten'

#
# Simulates CROC signal shape in IT modules
#
class UpgradeWorkflow_ITSignalShape(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Digi' in step:
            stepDict[stepName][k] = merge([{'--customise': 'SimTracker/SiPhase2Digitizer/customizeForPhase2TrackerSignalShape.customizeSiPhase2ITSignalShape'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return 'Run4' in key
# define several of them
upgradeWFs['ITSignalShape'] = UpgradeWorkflow_ITSignalShape(
    steps =  [
        'Digi',
        'DigiTrigger',
    ],
    PU =  [
        'Digi',
        'DigiTrigger',
    ],
    suffix = '_ITSignalShape',
    offset = 0.141
)

# Specifying explicitly the --filein is not nice but that was the
# easiest way to "skip" the output of step2 (=premixing stage1) for
# filein (as it goes to pileup_input). It works (a bit accidentally
# though) also for "-i all" because in that case the --filein for DAS
# input is after this one in the list of command line arguments to
# cmsDriver, and gets then used in practice.
digiPremixLocalPileup = {
    "--filein": "file:step1.root",
    "--pileup_input": "file:step2.root"
}

# for premix
class UpgradeWorkflowPremix(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        # just copy steps
        stepDict[stepName][k] = merge([stepDict[step][k]])
    def setupPU_(self, step, stepName, stepDict, k, properties):
        # fastsim version
        if 'FS' in k:
            # setup for stage 1 fastsim
            if "Gen" in stepName:
                stepNamePmx = stepName.replace('Gen','Premix')
                if not stepNamePmx in stepDict: stepDict[stepNamePmx] = {}
                stepDict[stepNamePmx][k] = merge([
                    {
                        '-s': 'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid',
                        '--fast':'',
                        '--datatier': 'PREMIX',
                        '--eventcontent': 'PREMIX',
                        '--procModifiers': 'premix_stage1'
                    },
                    stepDict[stepName][k]
                ])
                if "ProdLike" in self.suffix:
                    # todo
                    pass
            # setup for stage 2 fastsim
            elif "FastSimRun3" in step:
                # go back to non-PU step version
                d = merge([stepDict[self.getStepName(step)][k]])
                if d is None: return
                tmpsteps = []
                for s in d["-s"].split(","):
                    if s == "DIGI" or "DIGI:" in s:
                        tmpsteps.extend([s, "DATAMIX"])
                    else:
                        tmpsteps.append(s)
                d = merge([{"-s"             : ",".join(tmpsteps),
                            "--datamix"      : "PreMix"},
                           d])
                if "--procModifiers" in d:
                    d["--procModifiers"] += ",premix_stage2"
                else:
                    d["--procModifiers"] = "premix_stage2"
                # for combined stage1+stage2
                if "_PMXS1S2" in self.suffix:
                    d = merge([digiPremixLocalPileup, d])
                stepDict[stepName][k] = d
            elif "HARVESTFastRun3" in step:
                # increment input step number
                stepDict[stepName][k] = merge([{'--filein':'file:step3_inDQM.root'},stepDict[stepName][k]])
        else:
            # setup for stage 1
            if "GenSim" in stepName:
                stepNamePmx = stepName.replace('GenSim','Premix')
                if not stepNamePmx in stepDict: stepDict[stepNamePmx] = {}
                stepDict[stepNamePmx][k] = merge([
                    {
                        '-s': 'GEN,SIM,DIGI:pdigi_valid',
                        '--datatier': 'PREMIX',
                        '--eventcontent': 'PREMIX',
                        '--procModifiers': 'premix_stage1'
                    },
                    stepDict[stepName][k]
                ])
                if "ProdLike" in self.suffix:
                    stepDict[stepNamePmx][k] = merge([{'-s': 'GEN,SIM,DIGI'},stepDict[stepNamePmx][k]])
            # setup for stage 2
            elif "Digi" in step or "Reco" in step:
                # go back to non-PU step version
                d = merge([stepDict[self.getStepName(step)][k]])
                if d is None: return
                if "Digi" in step:
                    tmpsteps = []
                    for s in d["-s"].split(","):
                        if s == "DIGI" or "DIGI:" in s:
                            tmpsteps.extend([s, "DATAMIX"])
                        else:
                            tmpsteps.append(s)
                    d = merge([{"-s"             : ",".join(tmpsteps),
                                "--datamix"      : "PreMix",
                                "--procModifiers": "premix_stage2"},
                               d])
                    # for combined stage1+stage2
                    if "_PMXS1S2" in self.suffix:
                        d = merge([digiPremixLocalPileup, d])
                elif "Reco" in step:
                    if "--procModifiers" in d:
                        d["--procModifiers"] += ",premix_stage2"
                    else:
                        d["--procModifiers"] = "premix_stage2"
                stepDict[stepName][k] = d
            # separate nano step now only used in ProdLike workflows for Run3/Phase2
            elif "Nano"==step:
                # go back to non-PU step version
                d = merge([stepDict[self.getStepName(step)][k]])
                if "_PMXS1S2" in self.suffix and "--filein" in d:
                    filein = d["--filein"]
                    m = re.search("step(?P<ind>\\d+)", filein)
                    if m:
                        d["--filein"] = filein.replace(m.group(), "step%d"%(int(m.group("ind"))+1))
                stepDict[stepName][k] = d
                # run2/3 WFs use Nano (not NanoPU) in PU WF
                stepDict[self.getStepName(step)][k] = merge([d])
    def condition(self, fragment, stepList, key, hasHarvest):
        if not 'PU' in key:
            return False
        if not any(y in key for y in ['2022', '2023', '2024', '2025', 'Run4']):
            return False
        if self.suffix.endswith("S1"):
            return "NuGun" in fragment
        return True
    def workflow_(self, workflows, num, fragment, stepList, key):
        fragmentTmp = fragment
        if self.suffix.endswith("S1"):
            fragmentTmp = 'PREMIXUP' + key[2:].replace("PU", "").replace("Design", "") + '_PU25'
        super(UpgradeWorkflowPremix,self).workflow_(workflows, num, fragmentTmp, stepList, key)
# Premix stage1
upgradeWFs['PMXS1'] = UpgradeWorkflowPremix(
    steps = [
    ],
    PU = [
        'Gen',
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
    ],
    suffix = '_PMXS1',
    offset = 0.97,
)
# Premix stage2
upgradeWFs['PMXS2'] = UpgradeWorkflowPremix(
    steps = [],
    PU = [
        'Digi',
        'DigiTrigger',
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'Nano',
        'FastSimRun3',
        'HARVESTFastRun3',
    ],
    suffix = '_PMXS2',
    offset = 0.98,
)
# Premix combined stage1+stage2
upgradeWFs['PMXS1S2'] = UpgradeWorkflowPremix(
    steps = [],
    PU = [
        'Gen',
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'Nano',
        'FastSimRun3',
        'HARVESTFastRun3',
    ],
    suffix = '_PMXS1S2',
    offset = 0.99,
)
# Alternative version of above w/ less PU for PR tests
class UpgradeWorkflowAdjustPU(UpgradeWorkflowPremix):
    def setupPU_(self, step, stepName, stepDict, k, properties):
        # adjust first, so it gets copied into new Premix step
        if '--pileup' in stepDict[stepName][k]:
            stepDict[stepName][k]['--pileup'] = 'AVE_50_BX_25ns_m3p3'
        super(UpgradeWorkflowAdjustPU,self).setupPU_(step, stepName, stepDict, k, properties)
    def condition(self, fragment, stepList, key, hasHarvest):
        # restrict to phase2
        return super(UpgradeWorkflowAdjustPU,self).condition(fragment, stepList, key, hasHarvest) and 'Run4' in key
upgradeWFs['PMXS1S2PR'] = UpgradeWorkflowAdjustPU(
    steps = [],
    PU = [
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'Nano',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
    ],
    suffix = '_PMXS1S2PR',
    offset = 0.999,
)

class UpgradeWorkflowPremixProdLike(UpgradeWorkflowPremix,UpgradeWorkflow_ProdLike):
    def setup_(self, step, stepName, stepDict, k, properties):
        # copy steps, then apply specializations
        UpgradeWorkflowPremix.setup_(self, step, stepName, stepDict, k, properties)
        UpgradeWorkflow_ProdLike.setup_(self, step, stepName, stepDict, k, properties)
        #
        if 'Digi' in step:
            d = merge([stepDict[self.getStepName(step)][k]])
            tmpsteps = []
            for s in d["-s"].split(","):
                if "DIGI:pdigi_valid" in s:
                    tmpsteps.append("DIGI")
                else:
                    tmpsteps.append(s)
            d = merge([{"-s" : ",".join(tmpsteps),
                        "--eventcontent": "PREMIXRAW"},
                       d])
            stepDict[stepName][k] = d
    def condition(self, fragment, stepList, key, hasHarvest):
        # use both conditions
        return UpgradeWorkflowPremix.condition(self, fragment, stepList, key, hasHarvest) and UpgradeWorkflow_ProdLike.condition(self, fragment, stepList, key, hasHarvest)
# premix stage2
upgradeWFs['PMXS2ProdLike'] = UpgradeWorkflowPremixProdLike(
    steps = [],
    PU = [
        'Digi',
        'DigiTrigger',
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'Nano',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'MiniAOD',
        'ALCA',
    ],
    suffix = '_PMXS2ProdLike',
    offset = 0.9821,
)
# premix combined stage1+stage2
upgradeWFs['PMXS1S2ProdLike'] = UpgradeWorkflowPremixProdLike(
    steps = [],
    PU = [
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'Nano',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'MiniAOD',
        'ALCA',
    ],
    suffix = '_PMXS1S2ProdLike',
    offset = 0.9921,
)

class UpgradeWorkflow_Run3FStrackingOnly(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'HARVESTFastRun3' in step:
            stepDict[stepName][k] = merge([{'-s':'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM',
                                            '--fast':'',
                                            '--era':'Run3_FastSim',
                                            '--filein':'file:step1_inDQM.root'}, stepDict[step][k]])
        else:
            stepDict[stepName][k] = merge([stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and ('FS' in key)
upgradeWFs['Run3FStrackingOnly'] = UpgradeWorkflow_Run3FStrackingOnly(
    steps = [
        'Gen',
        'FastSimRun3',
        'HARVESTFastRun3'
    ],
    PU = [
        'FastSimRun3',
        'HARVESTFastRun3'
    ],
    suffix = '_Run3FSTrackingOnly',
    offset = 0.302,
)

class UpgradeWorkflow_Run3FSMBMixing(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Gen' in step and 'GenOnly' not in step:
            stepDict[stepName][k] = merge([{'-s':'GEN,SIM,RECOBEFMIX',
                                            '--fast':'',
                                            '--era':'Run3_FastSim',
                                            '--eventcontent':'FASTPU',
                                            '--datatier':'GEN-SIM-RECO',
                                            '--relval':'27000,3000'}, stepDict[step][k]])
        else:
            stepDict[stepName][k] = None
    def condition(self, fragment, stepList, key, hasHarvest):
        return ('FS' in key) and fragment=="MinBias_14TeV"
upgradeWFs['Run3FSMBMixing'] = UpgradeWorkflow_Run3FSMBMixing(
    steps = [
        'Gen',
        'FastSimRun3',
        'HARVESTFastRun3'
    ],
    PU = [],
    suffix = '_Run3FSMBMixing',
    offset = 0.303,
)


class UpgradeWorkflow_DD4hep(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Phase2' in stepDict[step][k]['--era']:
            dd4hepGeom="DD4hep"
            dd4hepGeom+=stepDict[step][k]['--geometry']
            stepDict[stepName][k] = merge([{'--geometry' : dd4hepGeom, '--procModifiers': 'dd4hep'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return ('Run4' in key) and ('FS' not in key)
upgradeWFs['DD4hep'] = UpgradeWorkflow_DD4hep(
    steps = [
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'ALCA',
    ],
    PU = [],
    suffix = '_DD4hep',
    offset = 0.911,
)
upgradeWFs['DD4hep'].allowReuse = False

#This workflow is now obsolete, it becomes default for Run-3.
#Keep it for future use in Phase-2, then delete
class UpgradeWorkflow_DD4hepDB(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Run3' in stepDict[step][k]['--era'] and 'Fast' not in stepDict[step][k]['--era']:
            stepDict[stepName][k] = merge([{'--conditions': 'auto:phase1_2022_realistic', '--geometry': 'DB:Extended'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and '2022' in key and 'FS' not in key
upgradeWFs['DD4hepDB'] = UpgradeWorkflow_DD4hepDB(
    steps = [
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'ALCA',
    ],
    PU = [],
    suffix = '_DD4hepDB',
    offset = 0.912,
)
upgradeWFs['DD4hepDB'].allowReuse = False

class UpgradeWorkflow_DDDDB(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        the_era = stepDict[step][k]['--era']
        exclude = ['2025','2024','2023','Fast','Pb']
        if 'Run3' in the_era and not any(e in the_era for e in exclude):
            # retain any other eras
            tmp_eras = the_era.split(',')
            tmp_eras[tmp_eras.index("Run3")] = 'Run3_DDD'
            tmp_eras = ','.join(tmp_eras)
            stepDict[stepName][k] = merge([{'--conditions': 'auto:phase1_2022_realistic_ddd', '--geometry': 'DB:Extended', '--era': tmp_eras}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and '2022' in key and 'FS' not in key and "HI" not in key
upgradeWFs['DDDDB'] = UpgradeWorkflow_DDDDB(
    steps = [
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'ALCA',
    ],
    PU = [],
    suffix = '_DDDDB',
    offset = 0.914,
)
upgradeWFs['DDDDB'].allowReuse = False

class UpgradeWorkflow_SonicTriton(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = merge([{'--procModifiers': 'allSonicTriton'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return ((fragment=='TTbar_13' or fragment=='TTbar_14TeV') and '2022' in key) \
            or (fragment=='TTbar_14TeV' and 'Run4' in key)
upgradeWFs['SonicTriton'] = UpgradeWorkflow_SonicTriton(
    steps = [
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'ALCA',
    ],
    PU = [
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
        'RecoNanoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
        'HARVESTNano',
        'HARVESTNanoFakeHLT',
        'ALCA',
    ],
    suffix = '_SonicTriton',
    offset = 0.9001,
)

class UpgradeWorkflow_Phase2_HeavyIon(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = merge([{'--procModifiers': 'phase2_pp_on_AA'}, stepDict[step][k]])
        if 'GenSim' in step:
            stepDict[stepName][k] = merge([{'--conditions': stepDict[step][k]["--conditions"].replace('_13TeV',''), '-n': 1}, stepDict[stepName][k]])
        elif 'Digi' in step:
            stepDict[stepName][k] = merge([{'-s': stepDict[step][k]["-s"].replace("DIGI:pdigi_valid","DIGI:pdigi_hi"), '--pileup': 'HiMixNoPU'}, stepDict[stepName][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=='HydjetQMinBias_5519GeV' and 'Run4' in key and 'PU' not in key

upgradeWFs['Phase2_HeavyIon'] = UpgradeWorkflow_Phase2_HeavyIon(
    steps = [
        'GenSimHLBeamSpot',
        'DigiTrigger',
        'RecoGlobal',
        'HARVESTGlobal',
        'ALCAPhase2'
    ],
    PU = [],
    suffix = '_hi',
    offset = 0.85,
)

# check for duplicates in offsets or suffixes
offsets  = [specialWF.offset for specialType,specialWF in upgradeWFs.items()]
suffixes = [specialWF.suffix for specialType,specialWF in upgradeWFs.items()]

dups = check_dups(offsets)
if len(dups)>0:
    raise ValueError("Duplicate special workflow offsets not allowed: "+','.join([str(x) for x in dups]))

dups = check_dups(suffixes)
if len(dups)>0:
    raise ValueError("Duplicate special workflow suffixes not allowed: "+','.join([str(x) for x in dups]))

upgradeProperties = {}

upgradeProperties[2017] = {
    '2017' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2017_realistic',
        'HLTmenu': '@relval2017',
        'Era' : 'Run2_2017',
        'ScenToRun' : ['GenSim','Digi','RecoFakeHLT','HARVESTFakeHLT','ALCA','Nano'],
    },
    '2017Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2017_design',
        'HLTmenu': '@relval2017',
        'Era' : 'Run2_2017',
        'BeamSpot': 'DBdesign',
        'ScenToRun' : ['GenSim','Digi','RecoFakeHLT','HARVESTFakeHLT'],
    },
    '2018' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2018_realistic',
        'HLTmenu': '@relval2018',
        'Era' : 'Run2_2018',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','Digi','RecoFakeHLT','HARVESTFakeHLT','ALCA','Nano'],
    },
    '2018Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2018_design',
        'HLTmenu': '@relval2018',
        'Era' : 'Run2_2018',
        'BeamSpot': 'DBdesign',
        'ScenToRun' : ['GenSim','Digi','RecoFakeHLT','HARVESTFakeHLT'],
    },
    '2022' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2022_realistic',
        'HLTmenu': '@relval2022',
        'Era' : 'Run3',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','Digi','RecoNanoFakeHLT','HARVESTNanoFakeHLT','ALCA'],
    },
    '2022Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2022_design',
        'HLTmenu': '@relval2022',
        'Era' : 'Run3',
        'BeamSpot': 'DBdesign',
        'ScenToRun' : ['GenSim','Digi','RecoNanoFakeHLT','HARVESTNanoFakeHLT'],
    },
    '2023' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2023_realistic',
        'HLTmenu': '@relval2023',
        'Era' : 'Run3_2023',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','Digi','RecoNanoFakeHLT','HARVESTNanoFakeHLT','ALCA'],
    },
    '2024' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2024_realistic',
        'HLTmenu': '@relval2024',
        'Era' : 'Run3_2024',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','Digi','RecoNanoFakeHLT','HARVESTNanoFakeHLT','ALCA'],
    },
    '2024HLTOnDigi' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2024_realistic',
        'HLTmenu': '@relval2024',
        'Era' : 'Run3',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','DigiNoHLT','HLTOnly','RecoNanoFakeHLT','HARVESTNanoFakeHLT','ALCA'],
    },
    '2022FS' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2022_realistic',
        'HLTmenu': '@relval2022',
        'Era' : 'Run3_FastSim',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['Gen','FastSimRun3','HARVESTFastRun3'],
    },
    '2022postEE' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2022_realistic_postEE',
        'HLTmenu': '@relval2022',
        'Era' : 'Run3',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','Digi','RecoNanoFakeHLT','HARVESTNanoFakeHLT','ALCA'],
    },
    '2023FS' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2023_realistic',
        'HLTmenu': '@relval2023',
        'Era' : 'Run3_2023_FastSim',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['Gen','FastSimRun3','HARVESTFastRun3'],
    },
    '2022HI' : {
        'Geom' : 'DB:Extended',
        'GT':'auto:phase1_2022_realistic_hi',
        'HLTmenu': '@fake2',
        'Era':'Run3_pp_on_PbPb',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','Digi','RecoNano','HARVESTNano','ALCA'],
    },
    '2022HIRP' : {
        'Geom' : 'DB:Extended',
        'GT':'auto:phase1_2022_realistic_hi',
        'HLTmenu': '@fake2',
        'Era':'Run3_pp_on_PbPb_approxSiStripClusters',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','Digi','RecoNano','HARVESTNano','ALCA'],
    },
    '2023HI' : {
        'Geom' : 'DB:Extended',
        'GT':'auto:phase1_2023_realistic_hi',
        'HLTmenu': '@fake2',
        'Era':'Run3_pp_on_PbPb',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','Digi','RecoNano','HARVESTNano','ALCA'],
    },
    '2023HIRP' : {
        'Geom' : 'DB:Extended',
        'GT':'auto:phase1_2023_realistic_hi',
        'HLTmenu': '@fake2',
        'Era':'Run3_pp_on_PbPb_approxSiStripClusters',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','Digi','RecoNano','HARVESTNano','ALCA'],
    },
    '2024GenOnly' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2024_realistic',
        'Era' : 'Run3',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['Gen'],
    },
    '2024SimOnGen' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2024_realistic',
        'HLTmenu': '@relval2024',
        'Era' : 'Run3',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['Gen','Sim','Digi','RecoNanoFakeHLT','HARVESTNanoFakeHLT','ALCA'],
    },
    '2024FS' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2024_realistic',
        'HLTmenu': '@relval2024',
        'Era' : 'Run3_FastSim',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['Gen','FastSimRun3','HARVESTFastRun3'],
    },
    '2025' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2025_realistic',
        'HLTmenu': '@relval2025',
        'Era' : 'Run3_2025',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','Digi','RecoNano','HARVESTNano','ALCA'],
    },
    '2025HLTOnDigi' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2025_realistic',
        'HLTmenu': '@relval2025',
        'Era' : 'Run3_2025',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['GenSim','DigiNoHLT','HLTOnly','RecoNano','HARVESTNano','ALCA'],
    },
    '2025GenOnly' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2025_realistic',
        'HLTmenu': '@relval2025',
        'Era' : 'Run3_2025',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['Gen'],
    },
    '2025SimOnGen' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2025_realistic',
        'HLTmenu': '@relval2025',
        'Era' : 'Run3_2025',
        'BeamSpot': 'DBrealistic',
        'ScenToRun' : ['Gen','Sim','Digi','RecoNano','HARVESTNano','ALCA'],
    },
    
}

# standard PU sequences
for key in list(upgradeProperties[2017].keys()):
    upgradeProperties[2017][key+'PU'] = deepcopy(upgradeProperties[2017][key])
    if 'FS' not in key:
        # update ScenToRun list
        scenToRun = upgradeProperties[2017][key+'PU']['ScenToRun']
        for idx,val in enumerate(scenToRun):
            # Digi -> DigiPU, Reco* -> Reco*PU, HARVEST* -> HARVEST*PU
            scenToRun[idx] += 'PU'*(val.startswith('Digi') or val.startswith('Reco') or val.startswith('HARVEST'))
        # remove ALCA
        upgradeProperties[2017][key+'PU']['ScenToRun'] = [foo for foo in scenToRun if foo != 'ALCA']
    else:
        upgradeProperties[2017][key+'PU']['ScenToRun'] = ['Gen','FastSimRun3PU','HARVESTFastRun3PU']

upgradeProperties['Run4'] = {
    'Run4D86' : {
        'Geom' : 'ExtendedRun4D86',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D88' : {
        'Geom' : 'ExtendedRun4D88',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D91' : {
        'Geom' : 'ExtendedRun4D91',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T30',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D92' : {
        'Geom' : 'ExtendedRun4D92',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D93' : {
        'Geom' : 'ExtendedRun4D93',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D94' : {
        'Geom' : 'ExtendedRun4D94',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C20I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D95' : {
        'Geom' : 'ExtendedRun4D95',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D96' : {
        'Geom' : 'ExtendedRun4D96',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D97' : {
        'Geom' : 'ExtendedRun4D97',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T25',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D98' : {
        'Geom' : 'ExtendedRun4D98',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T25',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D99' : {
        'Geom' : 'ExtendedRun4D99',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T25',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D100' : {
        'Geom' : 'ExtendedRun4D100',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T25',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D101' : {
        'Geom' : 'ExtendedRun4D101',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T25',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D102' : {
        'Geom' : 'ExtendedRun4D102',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D103' : {
        'Geom' : 'ExtendedRun4D103',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T25',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D104' : {
        'Geom' : 'ExtendedRun4D104',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C22I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D105' : {
        'Geom' : 'ExtendedRun4D105',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D106' : {
        'Geom' : 'ExtendedRun4D106',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C22I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D107' : {
        'Geom' : 'ExtendedRun4D107',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T25',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D108' : {
        'Geom' : 'ExtendedRun4D108',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D109' : {
        'Geom' : 'ExtendedRun4D109',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C22I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D110' : {
        'Geom' : 'ExtendedRun4D110',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
   'Run4D111' : {
        'Geom' : 'ExtendedRun4D111',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T36',
        'Era' : 'Phase2C22I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D112' : {
        'Geom' : 'ExtendedRun4D112',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T37',
        'Era' : 'Phase2C22I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D113' : {
        'Geom' : 'ExtendedRun4D113',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T38',
        'Era' : 'Phase2C22I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D114' : {
        'Geom' : 'ExtendedRun4D114',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D110GenOnly' : {
        'Geom' : 'ExtendedRun4D110',
        'BeamSpot' : 'DBrealisticHLLHC',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenHLBeamSpot'],
    },
    'Run4D110SimOnGen' : {
        'Geom' : 'ExtendedRun4D110',
        'HLTmenu': '@relvalRun4',
        'BeamSpot' : 'DBrealisticHLLHC',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenHLBeamSpot','Sim','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D115' : {
        'Geom' : 'ExtendedRun4D115',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C20I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    'Run4D116' : {
        'Geom' : 'ExtendedRun4D116',
        'HLTmenu': '@relvalRun4',
        'GT' : 'auto:phase2_realistic_T33',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
}

# standard PU sequences
for key in list(upgradeProperties['Run4'].keys()):
    upgradeProperties['Run4'][key+'PU'] = deepcopy(upgradeProperties['Run4'][key])
    upgradeProperties['Run4'][key+'PU']['ScenToRun'] = ['GenSimHLBeamSpot','DigiTriggerPU','RecoGlobalPU', 'HARVESTGlobalPU']

# for relvals
defaultDataSets = {}
for year in upgradeKeys:
    for key in upgradeKeys[year]:
        if 'PU' in key: continue
        defaultDataSets[key] = ''


class UpgradeFragment(object):
    def __init__(self, howMuch, dataset):
        self.howMuch = howMuch
        self.dataset = dataset

upgradeFragments = OrderedDict([
    ('FourMuPt_1_200_pythia8_cfi', UpgradeFragment(Kby(10,100),'FourMuPt1_200')),
    ('SingleElectronPt10_pythia8_cfi', UpgradeFragment(Kby(9,100),'SingleElectronPt10')),
    ('SingleElectronPt35_pythia8_cfi', UpgradeFragment(Kby(9,100),'SingleElectronPt35')),
    ('SingleElectronPt1000_pythia8_cfi', UpgradeFragment(Kby(9,50),'SingleElectronPt1000')),
    ('SingleGammaPt10_pythia8_cfi', UpgradeFragment(Kby(9,100),'SingleGammaPt10')),
    ('SingleGammaPt35_pythia8_cfi', UpgradeFragment(Kby(9,50),'SingleGammaPt35')),
    ('SingleMuPt1_pythia8_cfi', UpgradeFragment(Kby(25,100),'SingleMuPt1')),
    ('SingleMuPt10_Eta2p85_cfi', UpgradeFragment(Kby(9,100),'SingleMuPt10')),
    ('SingleMuPt100_Eta2p85_cfi', UpgradeFragment(Kby(9,100),'SingleMuPt100')),
    ('SingleMuPt1000_Eta2p85_cfi', UpgradeFragment(Kby(9,100),'SingleMuPt1000')),
    ('FourMuExtendedPt_1_200_pythia8_cfi', UpgradeFragment(Kby(10,100),'FourMuExtendedPt1_200')),
    ('TenMuExtendedE_0_200_pythia8_cfi', UpgradeFragment(Kby(10,100),'TenMuExtendedE_0_200')),
    ('DoubleElectronPt10Extended_pythia8_cfi', UpgradeFragment(Kby(9,100),'SingleElPt10Extended')),
    ('DoubleElectronPt35Extended_pythia8_cfi', UpgradeFragment(Kby(9,100),'SingleElPt35Extended')),
    ('DoubleElectronPt1000Extended_pythia8_cfi', UpgradeFragment(Kby(9,50),'SingleElPt1000Extended')),
    ('DoubleGammaPt10Extended_pythia8_cfi', UpgradeFragment(Kby(9,100),'SingleGammaPt10Extended')),
    ('DoubleGammaPt35Extended_pythia8_cfi', UpgradeFragment(Kby(9,50),'SingleGammaPt35Extended')),
    ('DoubleMuPt1Extended_pythia8_cfi', UpgradeFragment(Kby(25,100),'SingleMuPt1Extended')),
    ('DoubleMuPt10Extended_pythia8_cfi', UpgradeFragment(Kby(25,100),'SingleMuPt10Extended')),
    ('DoubleMuPt100Extended_pythia8_cfi', UpgradeFragment(Kby(9,100),'SingleMuPt100Extended')),
    ('DoubleMuPt1000Extended_pythia8_cfi', UpgradeFragment(Kby(9,100),'SingleMuPt1000Extended')),
    ('TenMuE_0_200_pythia8_cfi', UpgradeFragment(Kby(10,100),'TenMuE_0_200')),
    ('SinglePiE50HCAL_pythia8_cfi', UpgradeFragment(Kby(50,500),'SinglePiE50HCAL')),
    ('MinBias_13TeV_pythia8_TuneCUETP8M1_cfi', UpgradeFragment(Kby(90,100),'MinBias_13')),
    ('TTbar_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'TTbar_13')),
    ('ZEE_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,100),'ZEE_13')),
    ('QCD_Pt_600_800_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'QCD_Pt_600_800_13')),
    ('Wjet_Pt_80_120_14TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,100),'Wjet_Pt_80_120_14TeV')),
    ('Wjet_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'Wjet_Pt_3000_3500_14TeV')),
    ('LM1_sfts_14TeV_cfi', UpgradeFragment(Kby(9,100),'LM1_sfts_14TeV')),
    ('QCD_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'QCD_Pt_3000_3500_14TeV')),
    ('QCD_Pt_80_120_14TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,100),'QCD_Pt_80_120_14TeV')),
    ('H200ChargedTaus_Tauola_14TeV_cfi', UpgradeFragment(Kby(9,100),'Higgs200ChargedTaus_14TeV')),
    ('JpsiMM_14TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(66,100),'JpsiMM_14TeV')),
    ('TTbar_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(9,100),'TTbar_14TeV')),
    ('WE_14TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,100),'WE_14TeV')),
    ('ZTT_Tauola_All_hadronic_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(9,100),'ZTT_14TeV')),
    ('H130GGgluonfusion_14TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,100),'H130GGgluonfusion_14TeV')),
    ('PhotonJet_Pt_10_14TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,100),'PhotonJets_Pt_10_14TeV')),
    ('QQH1352T_Tauola_14TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,100),'QQH1352T_Tauola_14TeV')),
    ('MinBias_14TeV_pythia8_TuneCP5_cfi', UpgradeFragment(Kby(90,100),'MinBias_14TeV')),
    ('WToMuNu_14TeV_TuneCP5_pythia8_cfi', UpgradeFragment(Kby(9,100),'WToMuNu_14TeV')),
    ('ZMM_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(18,100),'ZMM_13')),
    ('QCDForPF_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(50,100),'QCD_FlatPt_15_3000HS_14')),
    ('DYToLL_M-50_14TeV_pythia8_cff', UpgradeFragment(Kby(9,100),'DYToLL_M_50_14TeV')),
    ('DYToTauTau_M-50_14TeV_pythia8_tauola_cff', UpgradeFragment(Kby(9,100),'DYtoTauTau_M_50_14TeV')),
    ('ZEE_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(9,100),'ZEE_14')),
    ('QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,100),'QCD_Pt_80_120_13')),
    ('H125GGgluonfusion_13TeV_TuneCP5_cfi', UpgradeFragment(Kby(9,50),'H125GGgluonfusion_13')),
    ('QCD_Pt20toInf_MuEnrichedPt15_14TeV_TuneCP5_cff', UpgradeFragment(Kby(19565, 217391),'QCD_Pt20toInfMuEnrichPt15_14')), # effi = 4.6e-4,  local=8.000e-04
    ('ZMM_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(18,100),'ZMM_14')),
    ('QCD_Pt15To7000_Flat_14TeV_TuneCP5_cff', UpgradeFragment(Kby(9,50),'QCD_Pt15To7000_Flat_14')),
    ('H125GGgluonfusion_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(9,50),'H125GGgluonfusion_14')),
    ('QCD_Pt_600_800_14TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'QCD_Pt_600_800_14')),
    ('UndergroundCosmicSPLooseMu_cfi', UpgradeFragment(Kby(9,50),'CosmicsSPLoose')),
    ('BeamHalo_13TeV_cfi', UpgradeFragment(Kby(9,50),'BeamHalo_13')),
    ('H200ChargedTaus_Tauola_13TeV_cfi', UpgradeFragment(Kby(9,50),'Higgs200ChargedTaus_13')),
    ('ADDMonoJet_13TeV_d3MD3_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'ADDMonoJet_d3MD3_13')),
    ('ZpMM_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'ZpMM_13')),
    ('QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'QCD_Pt_3000_3500_13')),
    ('WpM_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'WpM_13')),
    ('SingleNuE10_cfi', UpgradeFragment(Kby(9,50),'NuGun')),
    ('TTbarLepton_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'TTbarLepton_13')),
    ('WE_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'WE_13')),
    ('WM_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'WM_13')),
    ('ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'ZTT_13')),
    ('PhotonJet_Pt_10_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'PhotonJets_Pt_10_13')),
    ('QQH1352T_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'QQH1352T_13')),
    ('Wjet_Pt_80_120_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'Wjet_Pt_80_120_13')),
    ('Wjet_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'Wjet_Pt_3000_3500_13')),
    ('SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi', UpgradeFragment(Kby(9,50),'SMS-T1tttt_mGl-1500_mLSP-100_13')),
    ('QCDForPF_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(50,100),'QCD_FlatPt_15_3000HS_13')),
    ('PYTHIA8_PhiToMuMu_TuneCUETP8M1_13TeV_cff', UpgradeFragment(Kby(9,50),'PhiToMuMu_13')),
    ('RSKKGluon_m3000GeV_13TeV_TuneCUETP8M1_cff', UpgradeFragment(Kby(9,50),'RSKKGluon_m3000GeV_13')),
    ('ZpMM_2250_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'ZpMM_2250_13')),
    ('ZpEE_2250_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'ZpEE_2250_13')),
    ('ZpTT_1500_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'ZpTT_1500_13')),
    ('Upsilon1SToMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,50),'Upsilon1SToMuMu_13')),
    ('EtaBToJpsiJpsi_forSTEAM_TuneCUEP8M1_13TeV_cfi', UpgradeFragment(Kby(9,50),'EtaBToJpsiJpsi_13')),
    ('JpsiMuMu_Pt-8_forSTEAM_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(3100,100000),'JpsiMuMu_Pt-8')),
    ('BuMixing_BMuonFilter_forSTEAM_13TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(900,10000),'BuMixing_13')),
    ('HSCPstop_M_200_TuneCUETP8M1_13TeV_pythia8_cff', UpgradeFragment(Kby(9,50),'HSCPstop_M_200_13')),
    ('RSGravitonToGammaGamma_kMpl01_M_3000_TuneCUETP8M1_13TeV_pythia8_cfi', UpgradeFragment(Kby(9,50),'RSGravitonToGaGa_13')),
    ('WprimeToENu_M-2000_TuneCUETP8M1_13TeV-pythia8_cff', UpgradeFragment(Kby(9,50),'WpToENu_M-2000_13')),
    ('DisplacedSUSY_stopToBottom_M_800_500mm_TuneCP5_13TeV_pythia8_cff', UpgradeFragment(Kby(9,50),'DisplacedSUSY_stopToB_M_800_500mm_13')),
    ('TenE_E_0_200_pythia8_cfi', UpgradeFragment(Kby(9,100),'TenE_0_200')),
    ('FlatRandomPtAndDxyGunProducer_cfi', UpgradeFragment(Kby(9,100),'DisplacedMuonsDxy_0_500')),
    ('TenTau_E_15_500_pythia8_cfi', UpgradeFragment(Kby(9,100),'TenTau_15_500')),
    ('SinglePiPt25Eta1p7_2p7_cfi', UpgradeFragment(Kby(9,100),'SinglePiPt25Eta1p7_2p7')),
    ('SingleMuPt15Eta1p7_2p7_cfi', UpgradeFragment(Kby(9,100),'SingleMuPt15Eta1p7_2p7')),
    ('SingleGammaPt25Eta1p7_2p7_cfi', UpgradeFragment(Kby(9,100),'SingleGammaPt25Eta1p7_2p7')),
    ('SingleElectronPt15Eta1p7_2p7_cfi', UpgradeFragment(Kby(9,100),'SingleElectronPt15Eta1p7_2p7')),
    ('ZTT_All_hadronic_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(9,50),'ZTT_14')),
    ('CloseByParticle_Photon_ERZRanges_cfi', UpgradeFragment(Kby(9,100),'CloseByParticleGun')),
    ('CE_E_Front_300um_cfi', UpgradeFragment(Kby(9,100),'CloseByPGun_CE_E_Front_300um')),
    ('CE_E_Front_200um_cfi', UpgradeFragment(Kby(9,100),'CloseByPGun_CE_E_Front_200um')),
    ('CE_E_Front_120um_cfi', UpgradeFragment(Kby(9,100),'CloseByPGun_CE_E_Front_120um')),
    ('CE_H_Fine_300um_cfi', UpgradeFragment(Kby(9,100),'CloseByPGun_CE_H_Fine_300um')),
    ('CE_H_Fine_200um_cfi', UpgradeFragment(Kby(9,100),'CloseByPGun_CE_H_Fine_200um')),
    ('CE_H_Fine_120um_cfi', UpgradeFragment(Kby(9,100),'CloseByPGun_CE_H_Fine_120um')),
    ('CE_H_Coarse_Scint_cfi', UpgradeFragment(Kby(9,100),'CloseByPGun_CE_H_Coarse_Scint')),
    ('CE_H_Coarse_300um_cfi', UpgradeFragment(Kby(9,100),'CloseByPGun_CE_H_Coarse_300um')),
    ('SingleElectronFlatPt2To100_cfi', UpgradeFragment(Kby(9,100),'SingleEFlatPt2To100')),
    ('SingleMuFlatPt0p7To10_cfi', UpgradeFragment(Kby(9,100),'SingleMuFlatPt0p7To10')),
    ('SingleMuFlatPt2To100_cfi', UpgradeFragment(Kby(9,100),'SingleMuFlatPt2To100')),
    ('SingleGammaFlatPt8To150_cfi', UpgradeFragment(Kby(9,100),'SingleGammaFlatPt8To150')),
    ('SinglePiFlatPt0p7To10_cfi', UpgradeFragment(Kby(9,100),'SinglePiFlatPt0p7To10')),
    ('SingleTauFlatPt2To150_cfi', UpgradeFragment(Kby(9,100),'SingleTauFlatPt2To150')),
    ('FlatRandomPtAndDxyGunProducer_MuPt2To10_cfi', UpgradeFragment(Kby(9,100),'DisplacedMuPt2To10')),
    ('FlatRandomPtAndDxyGunProducer_MuPt10To30_cfi', UpgradeFragment(Kby(9,100),'DisplacedMuPt10To30')),
    ('FlatRandomPtAndDxyGunProducer_MuPt30To100_cfi', UpgradeFragment(Kby(9,100),'DisplacedMuPt30To100')),
    ('B0ToKstarMuMu_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(304,3030),'B0ToKstarMuMu_14TeV')), # 3.3%
    ('BsToEleEle_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(223,2222),'BsToEleEle_14TeV')), # 4.5%
    ('BsToJpsiGamma_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(2500,25000),'BsToJpsiGamma_14TeV')), # 0.4%
    ('BsToJpsiPhi_mumuKK_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(910,9090),'BsToJpsiPhi_mumuKK_14TeV')), # 1.1%
    ('BsToMuMu_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(313,3125),'BsToMuMu_14TeV')), # 3.2%
    ('BsToPhiPhi_KKKK_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(556,5555),'BsToPhiPhi_KKKK_14TeV')), # 1.8%
    ('TauToMuMuMu_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(18939,189393),'TauToMuMuMu_14TeV')), # effi = 5.280e-04
    ('BdToKstarEleEle_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(206,2061),'BdToKstarEleEle_14TeV')), #effi = 4.850e-02
    ('ZpTT_1500_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(9,50),'ZpTT_1500_14')),
    ('BuMixing_BMuonFilter_forSTEAM_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(900,10000),'BuMixing_14')),
    ('Upsilon1SToMuMu_forSTEAM_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(9,50),'Upsilon1SToMuMu_14')),
    ('TenTau_E_15_500_Eta3p1_pythia8_cfi', UpgradeFragment(Kby(9,100),'TenTau_15_500_Eta3p1')),
    ('QCD_Pt_1800_2400_14TeV_TuneCP5_cfi', UpgradeFragment(Kby(9,50), 'QCD_Pt_1800_2400_14')),
    ('DisplacedSUSY_stopToBottom_M_800_500mm_TuneCP5_14TeV_pythia8_cff', UpgradeFragment(Kby(9,50),'DisplacedSUSY_14TeV')),
    ('GluGluTo2Jets_M_300_2000_14TeV_Exhume_cff',UpgradeFragment(Kby(9,100),'GluGluTo2Jets_14TeV')),
    ('TTbarToDilepton_mt172p5_TuneCP5_14TeV_pythia8_cfi',UpgradeFragment(Kby(9,50),'TTbarToDilepton_14TeV')),
    ('QQToHToTauTau_mh125_TuneCP5_14TeV_pythia8_cfi',UpgradeFragment(Kby(9,50),'QQToHToTauTau_14TeV')),
    ('ZpToEE_m6000_TuneCP5_14TeV_pythia8_cfi',UpgradeFragment(Kby(9,50),'ZpToEE_m6000_14TeV')),
    ('ZpToMM_m6000_TuneCP5_14TeV_pythia8_cfi',UpgradeFragment(Kby(9,50),'ZpToMM_m6000_14TeV')),
    ('SMS-T1tttt_mGl-1500_mLSP-100_TuneCP5_14TeV_pythia8_cfi',UpgradeFragment(Kby(9,50),'SMS-T1tttt_14TeV')),
    ('VBFHZZ4Nu_TuneCP5_14TeV_pythia8_cfi',UpgradeFragment(Kby(9,50),'VBFHZZ4Nu_14TeV')),
    ('EtaBToJpsiJpsi_14TeV_TuneCP5_pythia8_cfi',UpgradeFragment(Kby(9,50),'EtaBToJpsiJpsi_14TeV')),
    ('WToLNu_14TeV_TuneCP5_pythia8_cfi',UpgradeFragment(Kby(21,50),'WToLNu_14TeV')),
    ('WprimeToLNu_M2000_14TeV_TuneCP5_pythia8_cfi',UpgradeFragment(Kby(21,50),'WprimeToLNu_M2000_14TeV')),
    ('DoubleMuFlatPt1p5To8_cfi', UpgradeFragment(Kby(9,100),'SingleMuFlatPt1p5To8')),
    ('DoubleElectronFlatPt1p5To8_cfi', UpgradeFragment(Kby(9,100),'SingleElectronFlatPt1p5To8')),
    ('DoubleMuFlatPt1p5To8Dxy100GunProducer_cfi', UpgradeFragment(Kby(9,100),'DisplacedMuPt1p5To8Dxy100')),
    ('DoubleMuFlatPt2To100Dxy100GunProducer_cfi', UpgradeFragment(Kby(9,100),'DisplacedMuPt2To100Dxy100')),
    ('BuToJPsiPrimeKToJPsiPiPiK_14TeV_TuneCP5_pythia8_cfi', UpgradeFragment(Kby(223,2222),'BuToJPsiPrimeKToJPsiPiPiK_14TeV')), # 5.7%
    ('Psi2SToJPsiPiPi_14TeV_TuneCP5_pythia8_cfi', UpgradeFragment(Kby(45,500),'Psi2SToJPsiPiPi_14TeV')), # 24.6%
    ('XiMinus_13p6TeV_SoftQCDInel_TuneCP5_cfi', UpgradeFragment(Kby(8000,90000),'XiMinus_13p6TeV')), #2.8%
    ('Chib1PToUpsilon1SGamma_MuFilter_TuneCP5_14TeV-pythia8_evtgen_cfi', UpgradeFragment(Kby(3600,36000),'Chib1PToUpsilon1SGamma_14TeV')), #2.8%
    ('ChicToJpsiGamma_MuFilter_TuneCP5_14TeV_pythia8_evtgen_cfi', UpgradeFragment(Kby(2000,20000),'ChicToJpsiGamma_14TeV')), #5%
    ('B0ToJpsiK0s_JMM_Filter_DGamma0_TuneCP5_13p6TeV-pythia8-evtgen_cfi',UpgradeFragment(Kby(18000,18000),'B0ToJpsiK0s_DGamma0_13p6TeV')), #2.7%
    ('DStarToD0Pi_D0ToKsPiPi_inclusive_SoftQCD_TuneCP5_13p6TeV-pythia8-evtgen',UpgradeFragment(Kby(38000,38000),'DStarToD0Pi_D0ToKsPiPi_13p6TeV')), #1.3%
    ('LbToJpsiLambda_JMM_Filter_DGamma0_TuneCP5_13p6TeV-pythia8-evtgen_cfi',UpgradeFragment(Mby(66,660000),'LbToJpsiLambda_DGamma0_13p6TeV')), #0.3%
    ('LbToJpsiXiK0sPi_JMM_Filter_DGamma0_TuneCP5_13p6TeV-pythia8-evtgen_cfi',UpgradeFragment(Mby(50,500000),'LbToJpsiXiK0sPr_DGamma0_13p6TeV')), #0.6%
    ('OmegaMinus_13p6TeV_SoftQCDInel_TuneCP5_cfi',UpgradeFragment(Mby(100,1000000),'OmegaMinus_13p6TeV')), #0.1%
    ('Hydjet_Quenched_MinBias_5020GeV_cfi', UpgradeFragment(U2000by1,'HydjetQMinBias_5020GeV')),
    ('Hydjet_Quenched_MinBias_5362GeV_cfi', UpgradeFragment(U2000by1,'HydjetQMinBias_5362GeV')),
    ('Hydjet_Quenched_MinBias_5519GeV_cfi', UpgradeFragment(U2000by1,'HydjetQMinBias_5519GeV')),
])
