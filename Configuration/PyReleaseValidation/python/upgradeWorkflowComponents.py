from copy import copy, deepcopy
from collections import OrderedDict
from .MatrixUtil import merge, Kby, Mby
import re

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
    '2021FS',
    '2021FSPU',
    '2021postEE',
    '2021postEEPU',
    '2023FS',
    '2023FSPU',
]

upgradeKeys[2026] = [
    '2026D86',
    '2026D86PU',
    '2026D88',
    '2026D88PU',
    '2026D91',
    '2026D91PU',
    '2026D92',
    '2026D92PU',
    '2026D93',
    '2026D93PU',
    '2026D94',
    '2026D94PU',
    '2026D95',
    '2026D95PU',
    '2026D96',
    '2026D96PU',
    '2026D97',
    '2026D97PU',
    '2026D98',
    '2026D98PU',
    '2026D99',
    '2026D99PU',
]

# pre-generation of WF numbers
numWFStart={
    2017: 10000,
    2026: 20000,
}
numWFSkip=200
# temporary measure to keep other WF numbers the same
numWFConflict = [[20400,20800], #D87
                 [21200,22000], #D89-D90
                 [50000,51000]]
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
        workflows[num+self.offset] = [ fragmentTmp, stepList ]
    def condition(self, fragment, stepList, key, hasHarvest):
        return False
    def preventReuse(self, stepName, stepDict, k):
        if "Sim" in stepName:
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
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'GenSimHLBeamSpotHGCALCloseBy',
        'Digi',
        'DigiTrigger',
        'HLTRun3',
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
            if 'Digi' in step:
                stepDict[stepName][k] = merge([{'-s': re.sub(',HLT.*', '', stepDict[step][k]['-s'])}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        if ('TTbar_14TeV' in fragment and '2021' == key):
            stepList.insert(stepList.index('Digi_DigiNoHLT_2021')+1, 'HLTRun3_2021')
        return ('TTbar_14TeV' in fragment and '2021' == key)
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
        result = (fragment=="TTbar_13" or fragment=="TTbar_14TeV") and not 'PU' in key and hasHarvest and self.condition_(fragment, stepList, key, hasHarvest)
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
        return ('2017' in key or '2018' in key or '2021' in key or '2026' in key) and ('FS' not in key)
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
        if 'Digi' in step: stepDict[stepName][k] = merge([self.step2, stepDict[step][k]])
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
    def condition_(self, fragment, stepList, key, hasHarvest):
        return ('2017' in key or '2021' in key or '2023' in key) and ('FS' not in key)
upgradeWFs['trackingMkFit'] = UpgradeWorkflow_trackingMkFit(
    steps = [
        'Digi',
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

#DeepCore seeding for JetCore iteration workflow
class UpgradeWorkflow_seedingDeepCore(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        # skip ALCA and Nano steps (but not RecoNano or HARVESTNano for Run3)
        if 'ALCA' in step or 'Nano'==step:
            stepDict[stepName][k] = None
        elif 'Reco' in step or 'HARVEST' in step: stepDict[stepName][k] = merge([{'--procModifiers': 'seedingDeepCore'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        result = (fragment=="QCD_Pt_1800_2400_14" or fragment=="TTbar_14TeV" ) and ('2021' in key or '2024' in key) and hasHarvest
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
        return '2021' in key
upgradeWFs['displacedRegional'] = UpgradeWorkflow_displacedRegional(
    steps = [
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'RecoNano',
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
        return (fragment=="TTbar_14TeV" or fragment=="SingleMuPt10Extended") and '2026' in key
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
    def __init__(self, reco = {}, harvest = {}, **kwargs):
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
        self.__reco = reco
        self.__harvest = harvest

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
        selected = [
            ('2021' in key and fragment == "TTbar_14TeV" and 'FS' not in key),
            ('2024' in key and fragment == "TTbar_14TeV"),
            ('2026' in key and fragment == "TTbar_14TeV")
        ]
        result = any(selected) and hasHarvest

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
        return (fragment=="TTbar_14TeV" or 'CloseByPGun_CE' in fragment) and '2026' in key
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
        return (fragment=="TTbar_14TeV" or 'CloseByPGun_CE' in fragment) and '2026' in key
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

class UpgradeWorkflow_ticl_v3(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'RecoGlobal' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        if 'HARVESTGlobal' in step:
            stepDict[stepName][k] = merge([self.step4, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="TTbar_14TeV" or 'CloseByP' in fragment or 'Eta1p7_2p7' in fragment) and '2026' in key
upgradeWFs['ticl_v3'] = UpgradeWorkflow_ticl_v3(
    steps = [
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    PU = [
        'RecoGlobal',
        'HARVESTGlobal'
    ],
    suffix = '_ticl_v3',
    offset = 0.203,
)
upgradeWFs['ticl_v3'].step3 = {'--procModifiers': 'ticl_v3'}
upgradeWFs['ticl_v3'].step4 = {'--procModifiers': 'ticl_v3'}


# Track DNN workflows
class UpgradeWorkflow_trackdnn(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = merge([{'--procModifiers': 'trackdnn'}, stepDict[step][k]])

    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and '2021' in key
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
        return (fragment=="TTbar_14TeV" or fragment=="QCD_FlatPt_15_3000HS_14") and '2021PU' in key

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
#   - 2018 conditions, TTbar
#   - 2018 conditions, Z->mumu
#   - 2022 conditions (labelled "2021"), TTbar
#   - 2022 conditions (labelled "2021"), Z->mumu
#   - 2023 conditions, TTbar
#   - 2023 conditions, Z->mumu
#   - 2026D88 conditions, TTbar
class PatatrackWorkflow(UpgradeWorkflow):
    def __init__(self, digi = {}, reco = {}, harvest = {}, **kwargs):
        # adapt the parameters for the UpgradeWorkflow init method
        super(PatatrackWorkflow, self).__init__(
            steps = [
                'Digi',
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
                'Nano',
                'ALCA',
                'ALCAPhase2'
            ],
            PU = [
                'Digi',
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
                'Nano',
                'ALCA',
                'ALCAPhase2'
            ],
            **kwargs)
        self.__digi = digi
        self.__reco = reco
        self.__reco.update({
            '--datatier':     'GEN-SIM-RECO,DQMIO',
            '--eventcontent': 'RECOSIM,DQM'
        })
        self.__harvest = harvest

    def condition(self, fragment, stepList, key, hasHarvest):
        # select only a subset of the workflows
        selected = [
            ('2018' in key and fragment == "TTbar_13"),
            ('2021' in key and fragment == "TTbar_14TeV" and 'FS' not in key),
            ('2023' in key and fragment == "TTbar_14TeV" and 'FS' not in key),
            ('2018' in key and fragment == "ZMM_13"),
            ('2021' in key and fragment == "ZMM_14" and 'FS' not in key),
            ('2023' in key and fragment == "ZMM_14" and 'FS' not in key),
            ('2026D88' in key and fragment == "TTbar_14TeV" and "PixelOnly" in self.suffix)
        ]
        result = any(selected) and hasHarvest

        return result

    def setup_(self, step, stepName, stepDict, k, properties):
        # skip ALCA and Nano steps (but not RecoNano or HARVESTNano for Run3)
        if 'ALCA' in step or 'Nano'==step:
            stepDict[stepName][k] = None
        elif 'Digi' in step:
            if self.__digi is None:
              stepDict[stepName][k] = None
            else:
              stepDict[stepName][k] = merge([self.__digi, stepDict[step][k]])
        elif 'Reco' in step:
            if self.__reco is None:
              stepDict[stepName][k] = None
            else:
              stepDict[stepName][k] = merge([self.__reco, stepDict[step][k]])
        elif 'HARVEST' in step:
            if self.__harvest is None:
              stepDict[stepName][k] = None
            else:
              stepDict[stepName][k] = merge([self.__harvest, stepDict[step][k]])

# Pixel-only quadruplets workflow running on CPU
#  - HLT on CPU
#  - Pixel-only reconstruction on CPU, with DQM and validation
#  - harvesting
upgradeWFs['PatatrackPixelOnlyCPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
        '--procModifiers': 'pixelNtupletFit'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'
    },
    suffix = 'Patatrack_PixelOnlyCPU',
    offset = 0.501,
)

# Pixel-only quadruplets workflow running on CPU or GPU
#  - HLT on GPU (optional)
#  - Pixel-only reconstruction on GPU (optional), with DQM and validation
#  - harvesting
upgradeWFs['PatatrackPixelOnlyGPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
        '--procModifiers': 'pixelNtupletFit,gpu'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'
    },
    suffix = 'Patatrack_PixelOnlyGPU',
    offset = 0.502,
)

# Pixel-only quadruplets workflow running on CPU and GPU
#  - HLT on GPU (required)
#  - Pixel-only reconstruction on both CPU and GPU, with DQM and validation for GPU-vs-CPU comparisons
#  - harvesting
upgradeWFs['PatatrackPixelOnlyGPUValidation'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'pixelNtupletFit,gpuValidation'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM',
        '--procModifiers': 'gpuValidation'
    },
    suffix = 'Patatrack_PixelOnlyGPU_Validation',
    offset = 0.503,
)

# Pixel-only quadruplets workflow running on CPU or GPU, trimmed down for benchmarking
#  - HLT on GPU (optional)
#  - Pixel-only reconstruction on GPU (optional)
upgradeWFs['PatatrackPixelOnlyGPUProfiling'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly',
        '--procModifiers': 'pixelNtupletFit,gpu',
        '--customise' : 'RecoTracker/Configuration/customizePixelOnlyForProfiling.customizePixelOnlyForProfilingGPUOnly'
    },
    harvest = None,
    suffix = 'Patatrack_PixelOnlyGPU_Profiling',
    offset = 0.504,
)

# Pixel-only triplets workflow running on CPU
#  - HLT on CPU
#  - Pixel-only reconstruction on CPU, with DQM and validation
#  - harvesting
upgradeWFs['PatatrackPixelOnlyTripletsCPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
        '--procModifiers': 'pixelNtupletFit',
        '--customise' : 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'
    },
    suffix = 'Patatrack_PixelOnlyTripletsCPU',
    offset = 0.505,
)

# Pixel-only triplets workflow running on CPU or GPU
#  - HLT on GPU (optional)
#  - Pixel-only reconstruction on GPU (optional), with DQM and validation
#  - harvesting
upgradeWFs['PatatrackPixelOnlyTripletsGPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
        '--procModifiers': 'pixelNtupletFit,gpu',
        '--customise': 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'
    },
    suffix = 'Patatrack_PixelOnlyTripletsGPU',
    offset = 0.506,
)

# Pixel-only triplets workflow running on CPU and GPU
#  - HLT on GPU (required)
#  - Pixel-only reconstruction on both CPU and GPU, with DQM and validation for GPU-vs-CPU comparisons
#  - harvesting
upgradeWFs['PatatrackPixelOnlyTripletsGPUValidation'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'pixelNtupletFit,gpuValidation',
        '--customise': 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM',
        '--procModifiers': 'gpuValidation',
    },
    suffix = 'Patatrack_PixelOnlyTripletsGPU_Validation',
    offset = 0.507,
)

# Pixel-only triplets workflow running on CPU or GPU, trimmed down for benchmarking
#  - HLT on GPU (optional)
#  - Pixel-only reconstruction on GPU (optional)
upgradeWFs['PatatrackPixelOnlyTripletsGPUProfiling'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly',
        '--procModifiers': 'pixelNtupletFit,gpu',
        '--customise': 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets,RecoTracker/Configuration/customizePixelOnlyForProfiling.customizePixelOnlyForProfilingGPUOnly'
    },
    harvest = None,
    suffix = 'Patatrack_PixelOnlyTripletsGPU_Profiling',
    offset = 0.508,
)

# ECAL-only workflow running on CPU
#  - HLT on CPU
#  - ECAL-only reconstruction on CPU, with DQM and validation
#  - harvesting
upgradeWFs['PatatrackECALOnlyCPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_ecalOnly,RECO:reconstruction_ecalOnly,VALIDATION:@ecalOnlyValidation,DQM:@ecalOnly',
    },
    harvest = {
        '-s': 'HARVESTING:@ecalOnlyValidation+@ecal'
    },
    suffix = 'Patatrack_ECALOnlyCPU',
    offset = 0.511,
)

# ECAL-only workflow running on CPU or GPU
#  - HLT on GPU (optional)
#  - ECAL-only reconstruction on GPU (optional), with DQM and validation
#  - harvesting
upgradeWFs['PatatrackECALOnlyGPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_ecalOnly,RECO:reconstruction_ecalOnly,VALIDATION:@ecalOnlyValidation,DQM:@ecalOnly',
        '--procModifiers': 'gpu'
    },
    harvest = {
        '-s': 'HARVESTING:@ecalOnlyValidation+@ecal'
    },
    suffix = 'Patatrack_ECALOnlyGPU',
    offset = 0.512,
)

# ECAL-only workflow running on CPU and GPU
#  - HLT on GPU (required)
#  - ECAL-only reconstruction on both CPU and GPU, with DQM and validation for GPU-vs-CPU comparisons
#  - harvesting
upgradeWFs['PatatrackECALOnlyGPUValidation'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_ecalOnly,RECO:reconstruction_ecalOnly,VALIDATION:@ecalOnlyValidation,DQM:@ecalOnly',
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'gpuValidation'
    },
    harvest = {
        '-s': 'HARVESTING:@ecalOnlyValidation+@ecal'
    },
    suffix = 'Patatrack_ECALOnlyGPU_Validation',
    offset = 0.513,
)

# ECAL-only workflow running on CPU or GPU, trimmed down for benchmarking
#  - HLT on GPU (optional)
#  - ECAL-only reconstruction on GPU (optional)
upgradeWFs['PatatrackECALOnlyGPUProfiling'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_ecalOnly,RECO:reconstruction_ecalOnly',
        '--procModifiers': 'gpu',
        '--customise' : 'RecoLocalCalo/Configuration/customizeEcalOnlyForProfiling.customizeEcalOnlyForProfilingGPUOnly'
    },
    harvest = None,
    suffix = 'Patatrack_ECALOnlyGPU_Profiling',
    offset = 0.514,
)

# HCAL-only workflow running on CPU
#  - HLT on CPU
#  - HCAL-only reconstruction on CPU, with DQM and validation
#  - harvesting
upgradeWFs['PatatrackHCALOnlyCPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnly,VALIDATION:@hcalOnlyValidation,DQM:@hcalOnly+@hcal2Only',
    },
    harvest = {
        '-s': 'HARVESTING:@hcalOnlyValidation+@hcalOnly+@hcal2Only'
    },
    suffix = 'Patatrack_HCALOnlyCPU',
    offset = 0.521,
)

# HCAL-only workflow running on CPU or GPU
#  - HLT on GPU (optional)
#  - HCAL-only reconstruction on GPU (optional), with DQM and validation
#  - harvesting
upgradeWFs['PatatrackHCALOnlyGPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnly,VALIDATION:@hcalOnlyValidation,DQM:@hcalOnly+@hcal2Only',
        '--procModifiers': 'gpu'
    },
    harvest = {
        '-s': 'HARVESTING:@hcalOnlyValidation+@hcalOnly+@hcal2Only'
    },
    suffix = 'Patatrack_HCALOnlyGPU',
    offset = 0.522,
)

# HCAL-only workflow running on CPU and GPU
#  - HLT on GPU (required)
#  - HCAL-only reconstruction on both CPU and GPU, with DQM and validation for GPU-vs-CPU comparisons
#  - harvesting
upgradeWFs['PatatrackHCALOnlyGPUValidation'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnly,VALIDATION:@hcalOnlyValidation,DQM:@hcalOnly+@hcal2Only',
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'gpuValidation'
    },
    harvest = {
        '-s': 'HARVESTING:@hcalOnlyValidation+@hcal'
    },
    suffix = 'Patatrack_HCALOnlyGPU_Validation',
    offset = 0.523,
)

# HCAL-only workflow running on CPU or GPU, trimmed down for benchmarking
#  - HLT on GPU (optional)
#  - HCAL-only reconstruction on GPU (optional)
upgradeWFs['PatatrackHCALOnlyGPUProfiling'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnly',
        '--procModifiers': 'gpu',
        '--customise' : 'RecoLocalCalo/Configuration/customizeHcalOnlyForProfiling.customizeHcalOnlyForProfilingGPUOnly'
    },
    harvest = None,
    suffix = 'Patatrack_HCALOnlyGPU_Profiling',
    offset = 0.524,
)

# Workflow running the Pixel quadruplets, ECAL and HCAL reconstruction on CPU
#  - HLT on CPU
#  - reconstruction on CPU, with DQM and validation
#  - harvesting
upgradeWFs['PatatrackAllCPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly+RawToDigi_ecalOnly+RawToDigi_hcalOnly,RECO:reconstruction_pixelTrackingOnly+reconstruction_ecalOnly+reconstruction_hcalOnly,VALIDATION:@pixelTrackingOnlyValidation+@ecalOnlyValidation+@hcalOnlyValidation,DQM:@pixelTrackingOnlyDQM+@ecalOnly+@hcalOnly+@hcal2Only',
        '--procModifiers': 'pixelNtupletFit'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM+@ecalOnlyValidation+@ecal+@hcalOnlyValidation+@hcalOnly+@hcal2Only'
    },
    suffix = 'Patatrack_AllCPU',
    offset = 0.581,
)

# Workflow running the Pixel quadruplets, ECAL and HCAL reconstruction on CPU or GPU
#  - HLT on GPU (optional)
#  - reconstruction on GPU (optional), with DQM and validation
#  - harvesting
upgradeWFs['PatatrackAllGPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly+RawToDigi_ecalOnly+RawToDigi_hcalOnly,RECO:reconstruction_pixelTrackingOnly+reconstruction_ecalOnly+reconstruction_hcalOnly,VALIDATION:@pixelTrackingOnlyValidation+@ecalOnlyValidation+@hcalOnlyValidation,DQM:@pixelTrackingOnlyDQM+@ecalOnly+@hcalOnly+@hcal2Only',
        '--procModifiers': 'pixelNtupletFit,gpu'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM+@ecalOnlyValidation+@ecal+@hcalOnlyValidation+@hcalOnly+@hcal2Only'
    },
    suffix = 'Patatrack_AllGPU',
    offset = 0.582,
)

# Workflow running the Pixel quadruplets, ECAL and HCAL reconstruction on CPU and GPU
#  - HLT on GPU (required)
#  - reconstruction on CPU and GPU, with DQM and validation for GPU-vs-CPU comparisons
#  - harvesting
upgradeWFs['PatatrackAllGPUValidation'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly+RawToDigi_ecalOnly+RawToDigi_hcalOnly,RECO:reconstruction_pixelTrackingOnly+reconstruction_ecalOnly+reconstruction_hcalOnly,VALIDATION:@pixelTrackingOnlyValidation+@ecalOnlyValidation+@hcalOnlyValidation,DQM:@pixelTrackingOnlyDQM+@ecalOnly+@hcalOnly+@hcal2Only',
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'pixelNtupletFit,gpuValidation'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM+@ecalOnlyValidation+@ecal+@hcalOnlyValidation+@hcalOnly+@hcal2Only',
        '--procModifiers': 'gpuValidation'
    },
    suffix = 'Patatrack_AllGPU_Validation',
    offset = 0.583,
)

# Workflow running the Pixel quadruplets, ECAL and HCAL reconstruction on CPU or GPU, trimmed down for benchmarking
#  - HLT on GPU (optional)
#  - minimal reconstruction on GPU (optional)
# FIXME workflow 0.584 to be implemented

# Workflow running the Pixel triplets, ECAL and HCAL reconstruction on CPU
#  - HLT on CPU
#  - reconstruction on CPU, with DQM and validation
#  - harvesting
upgradeWFs['PatatrackAllTripletsCPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly+RawToDigi_ecalOnly+RawToDigi_hcalOnly,RECO:reconstruction_pixelTrackingOnly+reconstruction_ecalOnly+reconstruction_hcalOnly,VALIDATION:@pixelTrackingOnlyValidation+@ecalOnlyValidation+@hcalOnlyValidation,DQM:@pixelTrackingOnlyDQM+@ecalOnly+@hcalOnly+@hcal2Only',
        '--procModifiers': 'pixelNtupletFit'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM+@ecalOnlyValidation+@ecal+@hcalOnlyValidation+@hcalOnly+@hcal2Only'
    },
    suffix = 'Patatrack_AllTripletsCPU',
    offset = 0.585,
)

# Workflow running the Pixel triplets, ECAL and HCAL reconstruction on CPU or GPU
#  - HLT on GPU (optional)
#  - reconstruction on GPU (optional), with DQM and validation
#  - harvesting
upgradeWFs['PatatrackAllTripletsGPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly+RawToDigi_ecalOnly+RawToDigi_hcalOnly,RECO:reconstruction_pixelTrackingOnly+reconstruction_ecalOnly+reconstruction_hcalOnly,VALIDATION:@pixelTrackingOnlyValidation+@ecalOnlyValidation+@hcalOnlyValidation,DQM:@pixelTrackingOnlyDQM+@ecalOnly+@hcalOnly+@hcal2Only',
        '--procModifiers': 'pixelNtupletFit,gpu'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM+@ecalOnlyValidation+@ecal+@hcalOnlyValidation+@hcalOnly+@hcal2Only'
    },
    suffix = 'Patatrack_AllTripletsGPU',
    offset = 0.586,
)

# Workflow running the Pixel triplets, ECAL and HCAL reconstruction on CPU and GPU
#  - HLT on GPU (required)
#  - reconstruction on CPU and GPU, with DQM and validation for GPU-vs-CPU comparisons
#  - harvesting
upgradeWFs['PatatrackAllTripletsGPUValidation'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'gpu'
    },
    reco = {
        '-s': 'RAW2DIGI:RawToDigi_pixelOnly+RawToDigi_ecalOnly+RawToDigi_hcalOnly,RECO:reconstruction_pixelTrackingOnly+reconstruction_ecalOnly+reconstruction_hcalOnly,VALIDATION:@pixelTrackingOnlyValidation+@ecalOnlyValidation+@hcalOnlyValidation,DQM:@pixelTrackingOnlyDQM+@ecalOnly+@hcalOnly+@hcal2Only',
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'pixelNtupletFit,gpuValidation'
    },
    harvest = {
        '-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM+@ecalOnlyValidation+@ecal+@hcalOnlyValidation+@hcalOnly+@hcal2Only',
        '--procModifiers': 'gpuValidation'
    },
    suffix = 'Patatrack_AllTripletsGPU_Validation',
    offset = 0.587,
)

# Workflow running the Pixel triplets, ECAL and HCAL reconstruction on CPU or GPU, trimmed down for benchmarking
#  - HLT on GPU (optional)
#  - minimal reconstruction on GPU (optional)
# FIXME workflow 0.588 to be implemented

# Workflow running the Pixel quadruplets, ECAL and HCAL reconstruction on CPU, together with the full offline reconstruction
#  - HLT on CPU
#  - reconstruction on CPU, with DQM and validation
#  - harvesting
upgradeWFs['PatatrackFullRecoCPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
    },
    reco = {
        # skip the @pixelTrackingOnlyValidation which cannot run together with the full reconstruction
        '-s': 'RAW2DIGI:RawToDigi+RawToDigi_pixelOnly,L1Reco,RECO:reconstruction+reconstruction_pixelTrackingOnly,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@pixelTrackingOnlyDQM',
        '--procModifiers': 'pixelNtupletFit'
    },
    harvest = {
        # skip the @pixelTrackingOnlyDQM harvesting
    },
    suffix = 'Patatrack_FullRecoCPU',
    offset = 0.591,
)

# Workflow running the Pixel quadruplets, ECAL and HCAL reconstruction on GPU (optional), together with the full offline reconstruction on CPU
#  - HLT on GPU (optional)
#  - reconstruction on GPU (optional), with DQM and validation
#  - harvesting
upgradeWFs['PatatrackFullRecoGPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        # skip the @pixelTrackingOnlyValidation which cannot run together with the full reconstruction
        '-s': 'RAW2DIGI:RawToDigi+RawToDigi_pixelOnly,L1Reco,RECO:reconstruction+reconstruction_pixelTrackingOnly,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@pixelTrackingOnlyDQM',
        '--procModifiers': 'pixelNtupletFit,gpu'
    },
    harvest = {
        # skip the @pixelTrackingOnlyDQM harvesting
    },
    suffix = 'Patatrack_FullRecoGPU',
    offset = 0.592,
)

# Workflow running the Pixel quadruplets, ECAL and HCAL reconstruction on CPU and GPU, together with the full offline reconstruction on CPU
#  - HLT on GPU (required)
#  - reconstruction on CPU and GPU, with DQM and validation for GPU-vs-CPU comparisons
#  - harvesting
upgradeWFs['PatatrackFullRecoGPUValidation'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'gpu'
    },
    reco = {
        # skip the @pixelTrackingOnlyValidation which cannot run together with the full reconstruction
        '-s': 'RAW2DIGI:RawToDigi+RawToDigi_pixelOnly,L1Reco,RECO:reconstruction+reconstruction_pixelTrackingOnly,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@pixelTrackingOnlyDQM',
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'pixelNtupletFit,gpuValidation'
    },
    harvest = {
        # skip the @pixelTrackingOnlyDQM harvesting
    },
    suffix = 'Patatrack_FullRecoGPU_Validation',
    offset = 0.593,
)

# Workflow running the Pixel triplets, ECAL and HCAL reconstruction on CPU, together with the full offline reconstruction
#  - HLT on CPU
#  - reconstruction on CPU, with DQM and validation
#  - harvesting
upgradeWFs['PatatrackFullRecoTripletsCPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
    },
    reco = {
        # skip the @pixelTrackingOnlyValidation which cannot run together with the full reconstruction
        '-s': 'RAW2DIGI:RawToDigi+RawToDigi_pixelOnly,L1Reco,RECO:reconstruction+reconstruction_pixelTrackingOnly,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@pixelTrackingOnlyDQM',
        '--procModifiers': 'pixelNtupletFit',
        '--customise' : 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets'
    },
    harvest = {
        # skip the @pixelTrackingOnlyDQM harvesting
    },
    suffix = 'Patatrack_FullRecoTripletsCPU',
    offset = 0.595,
)

# Workflow running the Pixel triplets, ECAL and HCAL reconstruction on GPU (optional), together with the full offline reconstruction on CPU
#  - HLT on GPU (optional)
#  - reconstruction on GPU (optional), with DQM and validation
#  - harvesting
upgradeWFs['PatatrackFullRecoTripletsGPU'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--procModifiers': 'gpu'
    },
    reco = {
        # skip the @pixelTrackingOnlyValidation which cannot run together with the full reconstruction
        '-s': 'RAW2DIGI:RawToDigi+RawToDigi_pixelOnly,L1Reco,RECO:reconstruction+reconstruction_pixelTrackingOnly,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@pixelTrackingOnlyDQM',
        '--procModifiers': 'pixelNtupletFit,gpu',
        '--customise': 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets'
    },
    harvest = {
        # skip the @pixelTrackingOnlyDQM harvesting
    },
    suffix = 'Patatrack_FullRecoTripletsGPU',
    offset = 0.596,
)

# Workflow running the Pixel triplets, ECAL and HCAL reconstruction on CPU and GPU, together with the full offline reconstruction on CPU
#  - HLT on GPU (required)
#  - reconstruction on CPU and GPU, with DQM and validation for GPU-vs-CPU comparisons
#  - harvesting
upgradeWFs['PatatrackFullRecoTripletsGPUValidation'] = PatatrackWorkflow(
    digi = {
        # the HLT menu is already set up for using GPUs if available and if the "gpu" modifier is enabled
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'gpu'
    },
    reco = {
        # skip the @pixelTrackingOnlyValidation which cannot run together with the full reconstruction
        '-s': 'RAW2DIGI:RawToDigi+RawToDigi_pixelOnly,L1Reco,RECO:reconstruction+reconstruction_pixelTrackingOnly,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@pixelTrackingOnlyDQM',
        '--accelerators': 'gpu-nvidia',
        '--procModifiers': 'pixelNtupletFit,gpuValidation',
        '--customise' : 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets'
    },
    harvest = {
        # skip the @pixelTrackingOnlyDQM harvesting
    },
    suffix = 'Patatrack_FullRecoTripletsGPU_Validation',
    offset = 0.597,
)


# end of Patatrack workflows

class UpgradeWorkflow_ProdLike(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'GenSimHLBeamSpot14' in step:
            stepDict[stepName][k] = merge([{'--eventcontent': 'RAWSIM', '--datatier': 'GEN-SIM'},stepDict[step][k]])
        elif 'Digi' in step and 'Trigger' not in step:
            stepDict[stepName][k] = merge([{'-s': 'DIGI,L1,DIGI2RAW,HLT:@relval2022', '--datatier':'GEN-SIM-RAW', '--eventcontent':'RAWSIM'}, stepDict[step][k]])
        elif 'DigiTrigger' in step: # for Phase-2
            stepDict[stepName][k] = merge([{'-s': 'DIGI,L1TrackTrigger,L1,DIGI2RAW,HLT:@fake2', '--datatier':'GEN-SIM-RAW', '--eventcontent':'RAWSIM'}, stepDict[step][k]])
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
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and ('2026' in key or '2021' in key)
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
        if stepDict[stepName][k] is not None and '--pileup' in stepDict[stepName][k]:
            stepDict[stepName][k]['--pileup'] = 'AVE_' + str(self.__fixedPU) + '_BX_25ns'
    def condition(self, fragment, stepList, key, hasHarvest):
        # lower PUs for Run3
        return (fragment=="TTbar_14TeV") and (('2026' in key) or ('2021' in key and self.__fixedPU<=100))

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

class UpgradeWorkflow_HLT75e33(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'HARVEST' in step:
            stepDict[stepName][k] = merge([{'--filein':'file:step3_inDQM.root'}, stepDict[step][k]])
        else:
            stepDict[stepName][k] = merge([stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and '2026' in key
upgradeWFs['HLT75e33'] = UpgradeWorkflow_HLT75e33(
    steps = [
        'GenSimHLBeamSpot14',
        'DigiTrigger',
        'RecoGlobal',
        'HLT75e33',
        'HARVESTGlobal',
    ],
    PU = [
        'GenSimHLBeamSpot14',
        'DigiTrigger',
        'RecoGlobal',
        'HLT75e33',
        'HARVESTGlobal',
    ],
    suffix = '_HLT75e33',
    offset = 0.75,
)

class UpgradeWorkflow_HLTwDIGI75e33(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'DigiTrigger' in step:
            stepDict[stepName][k] = merge([{'-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,DIGI2RAW,HLT:@relval2026'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and '2026' in key
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
upgradeWFs['Neutron'].neutronKeys = [x for x in upgradeKeys[2026] if 'PU' not in x]
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
        return fragment=="TTbar_14TeV" and '2026' in key

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

class UpgradeWorkflow_0T(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        myGT=stepDict[step][k]['--conditions']
        myGT+="_0T"
        stepDict[stepName][k] = merge([{'-n':'1','--magField':'0T','--conditions':myGT}, stepDict[step][k]])
    def setupPU_(self, step, stepName, stepDict, k, properties):
        # override '-n' setting from PUDataSets in relval_steps.py
        stepDict[stepName][k] = merge([{'-n':'1'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="TTbar_13" or fragment=="TTbar_14TeV") and ('2017' in key or '2018' in key or '2021' in key) and ('FS' not in key)
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
            stepDict[stepName][k] = merge([{'--customise': 'PhysicsTools/NanoAOD/custom_jme_cff.PrepJMECustomNanoAOD_MC'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="TTbar_13" or fragment=="TTbar_14TeV") and ('2017' in key or '2018' in key or '2021' in key) and ('FS' not in key)
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
        return '2026' in key
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

#
# Simulates Bias Rail in Phase-2 OT PS modules and X% random bad Strips
# in PS-s and SS sensors
#
class UpgradeWorkflow_OTInefficiency(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Digi' in step:
            stepDict[stepName][k] = merge([{'--customise': 'SimTracker/SiPhase2Digitizer/customizeForOTInefficiency.customizeSiPhase2OTInefficiency'+self.percent+'Percent'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and '2026' in key
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
        return '2026' in key
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
        # Increase the input file step number by one for Nano in combined stage1+stage2
        elif "Nano"==step:
            # go back to non-PU step version
            d = merge([stepDict[self.getStepName(step)][k]])
            if "--filein" in d:
                filein = d["--filein"]
                m = re.search("step(?P<ind>\d+)_", filein)
                if m:
                    d["--filein"] = filein.replace(m.group(), "step%d_"%(int(m.group("ind"))+1))
            stepDict[stepName][k] = d
            # run2/3 WFs use Nano (not NanoPU) in PU WF
            stepDict[self.getStepName(step)][k] = merge([d])
    def condition(self, fragment, stepList, key, hasHarvest):
        if not 'PU' in key:
            return False
        if not any(y in key for y in ['2021', '2023', '2024', '2026']):
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
    ],
    suffix = '_PMXS2',
    offset = 0.98,
)
# Premix combined stage1+stage2
upgradeWFs['PMXS1S2'] = UpgradeWorkflowPremix(
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
        return super(UpgradeWorkflowAdjustPU,self).condition(fragment, stepList, key, hasHarvest) and '2026' in key
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
        if 'Nano'==step:
            stepDict[stepName][k] = merge([{'--filein':'file:step5.root','-s':'NANO','--datatier':'NANOAODSIM','--eventcontent':'NANOEDMAODSIM'}, stepDict[step][k]])
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
        return ('2021FS' in key or '2023FS' in key)
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
        if 'Gen' in step:
            stepDict[stepName][k] = merge([{'-s':'GEN,SIM,RECOBEFMIX',
                                            '--fast':'',
                                            '--era':'Run3_FastSim',
                                            '--eventcontent':'FASTPU',
                                            '--datatier':'GEN-SIM-RECO',
                                            '--relval':'27000,3000'}, stepDict[step][k]])
        else:
            stepDict[stepName][k] = None
    def condition(self, fragment, stepList, key, hasHarvest):
        return ('2021FS' in key or '2023FS' in key) and fragment=="MinBias_14TeV"
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
        if 'Run3' in stepDict[step][k]['--era'] and 'Fast' not in stepDict[step][k]['--era']:
            if '2023' in stepDict[step][k]['--conditions']:
                stepDict[stepName][k] = merge([{'--geometry': 'DD4hepExtended2023'}, stepDict[step][k]])
            else:
                stepDict[stepName][k] = merge([{'--geometry': 'DD4hepExtended2021'}, stepDict[step][k]])
        elif 'Phase2' in stepDict[step][k]['--era']:
            dd4hepGeom="DD4hep"
            dd4hepGeom+=stepDict[step][k]['--geometry']
            stepDict[stepName][k] = merge([{'--geometry' : dd4hepGeom, '--procModifiers': 'dd4hep'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return ('2021' in key or '2023' in key or '2026' in key) and ('FS' not in key)
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
        return '2021' in key and 'FS' not in key
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
        if 'Run3' in stepDict[step][k]['--era']  and '2023' not in stepDict[step][k]['--era'] and 'Fast' not in stepDict[step][k]['--era']:
            # retain any other eras
            tmp_eras = stepDict[step][k]['--era'].split(',')
            tmp_eras[tmp_eras.index("Run3")] = 'Run3_DDD'
            tmp_eras = ','.join(tmp_eras)
            stepDict[stepName][k] = merge([{'--conditions': 'auto:phase1_2022_realistic_ddd', '--geometry': 'DB:Extended', '--era': tmp_eras}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return '2021' in key and 'FS' not in key
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
        return (fragment=='TTbar_13' and '2021' in key) \
            or (fragment=='TTbar_14TeV' and '2026' in key)
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

# check for duplicate offsets
offsets = [specialWF.offset for specialType,specialWF in upgradeWFs.items()]
seen = set()
dups = set(x for x in offsets if x in seen or seen.add(x))
if len(dups)>0:
    raise ValueError("Duplicate special workflow offsets not allowed: "+','.join([str(x) for x in dups]))

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
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSim','Digi','RecoFakeHLT','HARVESTFakeHLT'],
    },
    '2018' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2018_realistic',
        'HLTmenu': '@relval2018',
        'Era' : 'Run2_2018',
        'BeamSpot': 'Realistic25ns13TeVEarly2018Collision',
        'ScenToRun' : ['GenSim','Digi','RecoFakeHLT','HARVESTFakeHLT','ALCA','Nano'],
    },
    '2018Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2018_design',
        'HLTmenu': '@relval2018',
        'Era' : 'Run2_2018',
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSim','Digi','RecoFakeHLT','HARVESTFakeHLT'],
    },
    '2021' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2022_realistic',
        'HLTmenu': '@relval2022',
        'Era' : 'Run3',
        'BeamSpot': 'Realistic25ns13p6TeVEOY2022Collision',
        'ScenToRun' : ['GenSim','Digi','RecoNanoFakeHLT','HARVESTNanoFakeHLT','ALCA'],
    },
    '2021Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2022_design',
        'HLTmenu': '@relval2022',
        'Era' : 'Run3',
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSim','Digi','RecoNanoFakeHLT','HARVESTNanoFakeHLT'],
    },
    '2023' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2023_realistic',
        'HLTmenu': '@relval2023',
        'Era' : 'Run3_2023',
        'BeamSpot': 'Realistic25ns13p6TeVEarly2023Collision',
        'ScenToRun' : ['GenSim','Digi','RecoNano','HARVESTNano','ALCA'],
    },
    '2024' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2024_realistic',
        'HLTmenu': '@relval2023',
        'Era' : 'Run3',
        'BeamSpot': 'Realistic25ns13p6TeVEarly2022Collision',
        'ScenToRun' : ['GenSim','Digi','RecoNano','HARVESTNano','ALCA'],
    },
    '2021FS' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2022_realistic',
        'HLTmenu': '@relval2022',
        'Era' : 'Run3_FastSim',
        'BeamSpot': 'Realistic25ns13p6TeVEarly2022Collision',
        'ScenToRun' : ['Gen','FastSimRun3','HARVESTFastRun3'],
    },
    '2021postEE' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2022_realistic_postEE',
        'HLTmenu': '@relval2022',
        'Era' : 'Run3',
        'BeamSpot': 'Realistic25ns13p6TeVEarly2022Collision',
        'ScenToRun' : ['GenSim','Digi','RecoNanoFakeHLT','HARVESTNanoFakeHLT','ALCA'],
    },
    '2023FS' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2023_realistic',
        'HLTmenu': '@relval2023',
        'Era' : 'Run3_FastSim',
        'BeamSpot': 'Realistic25ns13p6TeVEarly2023Collision',
        'ScenToRun' : ['Gen','FastSimRun3','HARVESTFastRun3'],
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

upgradeProperties[2026] = {
    '2026D86' : {
        'Geom' : 'Extended2026D86',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    '2026D88' : {
        'Geom' : 'Extended2026D88',
        'HLTmenu': '@relval2026',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    '2026D91' : {
        'Geom' : 'Extended2026D91',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T30',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    '2026D92' : {
        'Geom' : 'Extended2026D92',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    '2026D93' : {
        'Geom' : 'Extended2026D93',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    '2026D94' : {
        'Geom' : 'Extended2026D94',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C20I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    '2026D95' : {
        'Geom' : 'Extended2026D95',
        'HLTmenu': '@relval2026',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    '2026D96' : {
        'Geom' : 'Extended2026D96',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    '2026D97' : {
        'Geom' : 'Extended2026D97',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T25',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    '2026D98' : {
        'Geom' : 'Extended2026D98',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T25',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
    '2026D99' : {
        'Geom' : 'Extended2026D99',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T25',
        'Era' : 'Phase2C17I13M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal', 'ALCAPhase2'],
    },
}

# standard PU sequences
for key in list(upgradeProperties[2026].keys()):
    upgradeProperties[2026][key+'PU'] = deepcopy(upgradeProperties[2026][key])
    upgradeProperties[2026][key+'PU']['ScenToRun'] = ['GenSimHLBeamSpot','DigiTriggerPU','RecoGlobalPU', 'HARVESTGlobalPU']

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
    ('B0ToJpsiK0s_JMM_Filter_DGamma0_TuneCP5_13p6TeV-pythia8-evtgen_cfi',UpgradeFragment(Kby(38000,38000),'B0ToJpsiK0s_DGamma0_13p6TeV')), #2.7%
    ('DStarToD0Pi_D0ToKsPiPi_inclusive_SoftQCD_TuneCP5_13p6TeV-pythia8-evtgen',UpgradeFragment(Kby(38000,38000),'DStarToD0Pi_D0ToKsPiPi_13p6TeV')), #1.3%
    ('LbToJpsiLambda_JMM_Filter_DGamma0_TuneCP5_13p6TeV-pythia8-evtgen_cfi',UpgradeFragment(Mby(66,660000),'LbToJpsiLambda_DGamma0_13p6TeV')), #0.3%
    ('LbToJpsiXiK0sPi_JMM_Filter_DGamma0_TuneCP5_13p6TeV-pythia8-evtgen_cfi',UpgradeFragment(Mby(50,500000),'LbToJpsiXiK0sPr_DGamma0_13p6TeV')), #0.6%
    ('OmegaMinus_13p6TeV_SoftQCDInel_TuneCP5_cfi',UpgradeFragment(Mby(100,1000000),'OmegaMinus_13p6TeV')), #0.1%
])
