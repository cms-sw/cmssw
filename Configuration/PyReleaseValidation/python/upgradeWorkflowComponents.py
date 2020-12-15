from copy import deepcopy
from collections import OrderedDict
import six
from .MatrixUtil import merge, Kby
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
]

upgradeKeys[2026] = [
    '2026D49',
    '2026D49PU',
    '2026D60',
    '2026D60PU',
    '2026D64',
    '2026D64PU',
    '2026D65',
    '2026D65PU',
    '2026D66',
    '2026D66PU',
    '2026D67',
    '2026D67PU',
    '2026D68',
    '2026D68PU',
    '2026D69',
    '2026D69PU',
    '2026D70',
    '2026D70PU',
    '2026D71',
    '2026D71PU',
    '2026D72',
    '2026D72PU',
    '2026D73',
    '2026D73PU',
    '2026D74',
    '2026D74PU',
]

# pre-generation of WF numbers
numWFStart={
    2017: 10000,
    2026: 20000,
}
numWFSkip=200
# temporary measure to keep other WF numbers the same
numWFConflict = [[20000,23200],[23600,28200],[28600,29800],[50000,51000]]
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
upgradeWFs = OrderedDict()

class UpgradeWorkflow_baseline(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        cust=properties.get('Custom', None)
        era=properties.get('Era', None)
        modifier=properties.get('ProcessModifier',None)
        if cust is not None: stepDict[stepName][k]['--customise']=cust
        if era is not None: stepDict[stepName][k]['--era']=era
        if modifier is not None: stepDict[stepName][k]['--procModifier']=modifier
    def condition(self, fragment, stepList, key, hasHarvest):
        return True
upgradeWFs['baseline'] = UpgradeWorkflow_baseline(
    steps =  [
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'RecoLocal',
        'Reco',
        'RecoFakeHLT',
        'RecoGlobal',
        'HARVEST',
        'HARVESTFakeHLT',
        'FastSim',
        'HARVESTFast',
        'HARVESTGlobal',
        'ALCA',
        'Nano',
        'MiniAOD',
    ],
    PU =  [
        'DigiTrigger',
        'RecoLocal',
        'RecoGlobal',
        'Digi',
        'Reco',
        'RecoFakeHLT',
        'HARVEST',
        'HARVESTFakeHLT',
        'HARVESTGlobal',
        'MiniAOD',
        'Nano',
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
            skipList = [s for s in stepList if (("ALCA" in s) or ("Nano" in s))]
            for skip in skipList:
                stepList.remove(skip)
        return result
    def condition_(self, fragment, stepList, key, hasHarvest):
        return True

class UpgradeWorkflow_trackingOnly(UpgradeWorkflowTracking):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step: stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step: stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM'}, stepDict[step][k]])
upgradeWFs['trackingOnly'] = UpgradeWorkflow_trackingOnly(
    steps = [
        'Reco',
        'HARVEST',
        'RecoGlobal',
        'HARVESTGlobal',
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
        'Reco',
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
        'Reco',
        'HARVEST',
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
        'Reco',
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
        return '2017' in key or '2018' in key or '2021' in key
upgradeWFs['pixelTrackingOnly'] = UpgradeWorkflow_pixelTrackingOnly(
    steps = [
        'Reco',
        'HARVEST',
        'RecoGlobal',
        'HARVESTGlobal',
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
        'Reco',
        'RecoGlobal',
    ],
    PU = [],
    suffix = '_trackingMkFit',
    offset = 0.7,
)
upgradeWFs['trackingMkFit'].step3 = {
    '--procModifiers': 'trackingMkFit'
}

#DeepCore seeding for JetCore iteration workflow
class UpgradeWorkflow_seedingDeepCore(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step or 'HARVEST' in step: stepDict[stepName][k] = merge([{'--procModifiers': 'seedingDeepCore'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        result = (fragment=="QCD_Pt_1800_2400_14") and ('2021' in key or '2024' in key) and hasHarvest
        if result:
            # skip ALCA and Nano
            skipList = [s for s in stepList if (("ALCA" in s) or ("Nano" in s))]
            for skip in skipList:
                stepList.remove(skip)
        return result
upgradeWFs['seedingDeepCore'] = UpgradeWorkflow_seedingDeepCore(
    steps = [
        'Reco',
        'HARVEST',
        'RecoGlobal',
        'HARVESTGlobal',
    ],
    PU = [],
    suffix = '_seedingDeepCore',
    offset = 0.17,
)

# Vector Hits workflows
class UpgradeWorkflow_vectorHits(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        stepDict[stepName][k] = merge([{'--procModifiers': 'vectorHits'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and '2026' in key
upgradeWFs['vectorHits'] = UpgradeWorkflow_vectorHits(
    steps = [
        'RecoGlobal',
    ],
    PU = [
        'RecoGlobal',
    ],
    suffix = '_vectorHits',
    offset = 0.9,
)

# Patatrack workflows
class UpgradeWorkflowPatatrack(UpgradeWorkflow):
    def condition(self, fragment, stepList, key, hasHarvest):
        is_2018_ttbar = ('2018' in key and fragment=="TTbar_13")
        is_2021_ttbar = ('2021' in key and fragment=="TTbar_14TeV")
        is_2018_zmumu = ('2018' in key and fragment=="ZMM_13")
        is_2021_zmumu = ('2021' in key and fragment=="ZMM_14")
        result = any((is_2018_ttbar, is_2021_ttbar, is_2018_zmumu, is_2021_zmumu)) and hasHarvest and self.condition_(fragment, stepList, key, hasHarvest)
        if result:
            # skip ALCA and Nano
            skipList = [s for s in stepList if (("ALCA" in s) or ("Nano" in s))]
            for skip in skipList:
                stepList.remove(skip)
        return result
    def condition_(self, fragment, stepList, key, hasHarvest):
        return True

class UpgradeWorkflowPatatrack_PixelOnlyCPU(UpgradeWorkflowPatatrack):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step:
            stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'}, stepDict[step][k]])

    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2018' in key or '2021' in key

upgradeWFs['PatatrackPixelOnlyCPU'] = UpgradeWorkflowPatatrack_PixelOnlyCPU(
    steps = [
        'Reco',
        'HARVEST',
        'RecoFakeHLT',
        'HARVESTFakeHLT',
        'RecoGlobal',
        'HARVESTGlobal',
    ],
    PU = [],
    suffix = 'Patatrack_PixelOnlyCPU',
    offset = 0.501,
)

upgradeWFs['PatatrackPixelOnlyCPU'].step3 = {
    '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
    '--datatier': 'GEN-SIM-RECO,DQMIO',
    '--eventcontent': 'RECOSIM,DQM',
    '--procModifiers': 'pixelNtupleFit'
}

class UpgradeWorkflowPatatrack_PixelOnlyGPU(UpgradeWorkflowPatatrack):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step:
            stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@pixelTrackingOnlyDQM'}, stepDict[step][k]])

    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2018' in key or '2021' in key

upgradeWFs['PatatrackPixelOnlyGPU'] = UpgradeWorkflowPatatrack_PixelOnlyGPU(
    steps = [
        'Reco',
        'HARVEST',
        'RecoFakeHLT',
        'HARVESTFakeHLT',
        'RecoGlobal',
        'HARVESTGlobal',
    ],
    PU = [],
    suffix = 'Patatrack_PixelOnlyGPU',
    offset = 0.502,
)

upgradeWFs['PatatrackPixelOnlyGPU'].step3 = {
    '-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,VALIDATION:@pixelTrackingOnlyValidation,DQM:@pixelTrackingOnlyDQM',
    '--datatier': 'GEN-SIM-RECO,DQMIO',
    '--eventcontent': 'RECOSIM,DQM',
    '--procModifiers': 'gpu'
}

class UpgradeWorkflowPatatrack_ECALOnlyCPU(UpgradeWorkflowPatatrack):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step:
            stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@ecalOnlyValidation+@ecal'}, stepDict[step][k]])

    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2018' in key or '2021' in key

upgradeWFs['PatatrackECALOnlyCPU'] = UpgradeWorkflowPatatrack_ECALOnlyCPU(
    steps = [
        'Reco',
        'HARVEST',
        'RecoFakeHLT',
        'HARVESTFakeHLT',
        'RecoGlobal',
        'HARVESTGlobal',
    ],
    PU = [],
    suffix = 'Patatrack_ECALOnlyCPU',
    offset = 0.511,
)

upgradeWFs['PatatrackECALOnlyCPU'].step3 = {
    '-s': 'RAW2DIGI:RawToDigi_ecalOnly,RECO:reconstruction_ecalOnly,VALIDATION:@ecalOnlyValidation,DQM:@ecalOnly',
    '--datatier': 'GEN-SIM-RECO,DQMIO',
    '--eventcontent': 'RECOSIM,DQM',
}

class UpgradeWorkflowPatatrack_ECALOnlyGPU(UpgradeWorkflowPatatrack):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step:
            stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@ecalOnlyValidation+@ecal'}, stepDict[step][k]])

    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2018' in key or '2021' in key

upgradeWFs['PatatrackECALOnlyGPU'] = UpgradeWorkflowPatatrack_ECALOnlyGPU(
    steps = [
        'Reco',
        'HARVEST',
        'RecoFakeHLT',
        'HARVESTFakeHLT',
        'RecoGlobal',
        'HARVESTGlobal',
    ],
    PU = [],
    suffix = 'Patatrack_ECALOnlyGPU',
    offset = 0.512,
)

upgradeWFs['PatatrackECALOnlyGPU'].step3 = {
    '-s': 'RAW2DIGI:RawToDigi_ecalOnly,RECO:reconstruction_ecalOnly,VALIDATION:@ecalOnlyValidation,DQM:@ecalOnly',
    '--datatier': 'GEN-SIM-RECO,DQMIO',
    '--eventcontent': 'RECOSIM,DQM',
    '--procModifiers': 'gpu'
}

class UpgradeWorkflowPatatrack_HCALOnlyCPU(UpgradeWorkflowPatatrack):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step:
            stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@hcalOnlyValidation+@hcalOnly+@hcal2Only'}, stepDict[step][k]])

    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2018' in key or '2021' in key

upgradeWFs['PatatrackHCALOnlyCPU'] = UpgradeWorkflowPatatrack_HCALOnlyCPU(
    steps = [
        'Reco',
        'HARVEST',
        'RecoFakeHLT',
        'HARVESTFakeHLT',
        'RecoGlobal',
        'HARVESTGlobal',
    ],
    PU = [],
    suffix = 'Patatrack_HCALOnlyCPU',
    offset = 0.521,
)

upgradeWFs['PatatrackHCALOnlyCPU'].step3 = {
    '-s': 'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnly,VALIDATION:@hcalOnlyValidation,DQM:@hcalOnly+@hcal2Only',
    '--datatier': 'GEN-SIM-RECO,DQMIO',
    '--eventcontent': 'RECOSIM,DQM',
}

class UpgradeWorkflowPatatrack_HCALOnlyGPU(UpgradeWorkflowPatatrack):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Reco' in step:
            stepDict[stepName][k] = merge([self.step3, stepDict[step][k]])
        elif 'HARVEST' in step:
            stepDict[stepName][k] = merge([{'-s': 'HARVESTING:@hcalOnlyValidation+@hcalOnly+@hcal2Only'}, stepDict[step][k]])

    def condition_(self, fragment, stepList, key, hasHarvest):
        return '2018' in key or '2021' in key

upgradeWFs['PatatrackHCALOnlyGPU'] = UpgradeWorkflowPatatrack_HCALOnlyGPU(
    steps = [
        'Reco',
        'HARVEST',
        'RecoFakeHLT',
        'HARVESTFakeHLT',
        'RecoGlobal',
        'HARVESTGlobal',
    ],
    PU = [],
    suffix = 'Patatrack_HCALOnlyGPU',
    offset = 0.522,
)

upgradeWFs['PatatrackHCALOnlyGPU'].step3 = {
    '-s': 'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnly,VALIDATION:@hcalOnlyValidation,DQM:@hcalOnly+@hcal2Only',
    '--datatier': 'GEN-SIM-RECO,DQMIO',
    '--eventcontent': 'RECOSIM,DQM',
    '--procModifiers': 'gpu'
}

# end of Patatrack workflows

class UpgradeWorkflow_ProdLike(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Digi' in step and 'Trigger' not in step:
            stepDict[stepName][k] = merge([{'-s': 'DIGI,L1,DIGI2RAW,HLT:@relval2021', '--datatier':'GEN-SIM-DIGI-RAW', '--eventcontent':'RAWSIM'}, stepDict[step][k]])
        elif 'Reco' in step:
            stepDict[stepName][k] = merge([{'-s': 'RAW2DIGI,L1Reco,RECO,RECOSIM', '--datatier':'AODSIM', '--eventcontent':'AODSIM'}, stepDict[step][k]])
        elif 'MiniAOD' in step:
            # the separate miniAOD step is used here
            stepDict[stepName][k] = deepcopy(stepDict[step][k])
        if 'ALCA' in step or 'HARVEST' in step:
            # remove step
            stepDict[stepName][k] = None
        if 'Nano' in step:
            stepDict[stepName][k] = merge([{'--filein':'file:step4.root'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and ('2026' in key or '2021' in key)
upgradeWFs['ProdLike'] = UpgradeWorkflow_ProdLike(
    steps = [
        'Digi',
        'Reco',
        'RecoGlobal',
        'HARVEST',
        'HARVESTGlobal',
        'MiniAOD',
        'ALCA',
        'Nano',
    ],
    PU = [
        'Digi',
        'Reco',
        'RecoGlobal',
        'HARVEST',
        'HARVESTGlobal',
        'MiniAOD',
        'ALCA',
        'Nano',
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
        'HARVEST',
        'ALCA',
    ],
    PU = [
        'Digi',
        'Reco',
        'HARVEST',
    ],
    suffix = '_heCollapse',
    offset = 0.6,
)

class UpgradeWorkflow_ecalDevel(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        # temporarily remove trigger & downstream steps
        mods = {'--era': stepDict[step][k]['--era']+',phase2_ecal_devel'}
        if 'Digi' in step:
            mods['-s'] = 'DIGI:pdigi_valid'
        stepDict[stepName][k] = merge([mods, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_14TeV" and '2026' in key
upgradeWFs['ecalDevel'] = UpgradeWorkflow_ecalDevel(
    steps = [
        'DigiTrigger',
        'RecoGlobal',
        'HARVESTGlobal',
    ],
    PU = [
        'DigiTrigger',
        'RecoGlobal',
        'HARVESTGlobal',
    ],
    suffix = '_ecalDevel',
    offset = 0.61,
)

class UpgradeWorkflow_0T(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        myGT=stepDict[step][k]['--conditions']
        myGT+="_0T"
        stepDict[stepName][k] = merge([{'-n':'1','--magField':'0T','--conditions':myGT}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return (fragment=="TTbar_13" or fragment=="TTbar_14TeV") and ('2017' in key or '2018' in key or '2021' in key)
upgradeWFs['0T'] = UpgradeWorkflow_0T(
    steps = [
        'GenSim',
        'Digi',
        'Reco',
        'HARVEST',
        'ALCA',
    ],
    PU = [
        'Digi',
        'Reco',
        'HARVEST',
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
    ],
    PU = [],
    suffix = '_ParkingBPH',
    offset = 0.8,
)

class UpgradeWorkflow_JMENano(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Nano' in step:
            stepDict[stepName][k] = merge([{'--customise': 'PhysicsTools/NanoAOD/custom_jme_cff.PrepJMECustomNanoAOD_MC'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return fragment=="TTbar_13" and ('2017' in key or '2018' in key)
upgradeWFs['JMENano'] = UpgradeWorkflow_JMENano(
    steps = [
        'Nano',
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
        return fragment=="TTbar_14TeV" and '2026' in key
# define several of them
upgradeWFs['Aging1000'] = UpgradeWorkflowAging(
    steps =  [
        'Digi',
        'DigiTrigger',
        'RecoLocal',
        'Reco',
        'RecoGlobal',
    ],
    PU =  [
        'Digi',
        'DigiTrigger',
        'RecoLocal',
        'Reco',
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
        elif "Nano" in step:
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
        'RecoGlobal',
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
        'RecoGlobal',
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
        'RecoGlobal',
        'Nano',
        'HARVEST',
        'HARVESTGlobal',
    ],
    suffix = '_PMXS1S2PR',
    offset = 0.999,
)

class UpgradeWorkflow_DD4hep(UpgradeWorkflow):
    def setup_(self, step, stepName, stepDict, k, properties):
        if 'Run3' in stepDict[step][k]['--era']:
            stepDict[stepName][k] = merge([{'--geometry': 'DD4hepExtended2021', '--procModifiers': 'dd4hep'}, stepDict[step][k]])
        elif 'Phase2' in stepDict[step][k]['--era']:
            dd4hepGeom="DD4hep"
            dd4hepGeom+=stepDict[step][k]['--geometry']
            stepDict[stepName][k] = merge([{'--geometry' : dd4hepGeom, '--procModifiers': 'dd4hep'}, stepDict[step][k]])
    def condition(self, fragment, stepList, key, hasHarvest):
        return ((fragment=='TTbar_13' or fragment=='ZMM_13' or fragment=='SingleMuPt10') and '2021' in key) \
            or ((fragment=='TTbar_14TeV' or fragment=='ZMM_14' or fragment=='SingleMuPt10') and '2026' in key)
upgradeWFs['DD4hep'] = UpgradeWorkflow_DD4hep(
    steps = [
        'GenSim',
        'GenSimHLBeamSpot',
        'GenSimHLBeamSpot14',
        'Digi',
        'DigiTrigger',
        'Reco',
        'RecoGlobal',
        'HARVEST',
        'HARVESTGlobal',
        'ALCA',
    ],
    PU = [],
    suffix = '_DD4hep',
    offset = 0.911,
)
upgradeWFs['DD4hep'].allowReuse = False

# check for duplicate offsets
offsets = [specialWF.offset for specialType,specialWF in six.iteritems(upgradeWFs)]
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
        'GT' : 'auto:phase1_2021_realistic',
        'HLTmenu': '@relval2021',
        'Era' : 'Run3',
        'BeamSpot': 'Run3RoundOptics25ns13TeVLowSigmaZ',
        'ScenToRun' : ['GenSim','Digi','Reco','HARVEST','ALCA'],
    },
    '2021Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2021_design',
        'HLTmenu': '@relval2021',
        'Era' : 'Run3',
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSim','Digi','Reco','HARVEST'],
    },
    '2023' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2023_realistic',
        'HLTmenu': '@relval2021',
        'Era' : 'Run3',
        'BeamSpot': 'Run3RoundOptics25ns13TeVLowSigmaZ',
        'ScenToRun' : ['GenSim','Digi','Reco','HARVEST','ALCA'],
    },
    '2024' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2024_realistic',
        'HLTmenu': '@relval2021',
        'Era' : 'Run3',
        'BeamSpot': 'Run3RoundOptics25ns13TeVLowSigmaZ',
        'ScenToRun' : ['GenSim','Digi','Reco','HARVEST','ALCA'],
    },
}

# standard PU sequences
for key in list(upgradeProperties[2017].keys()):
    upgradeProperties[2017][key+'PU'] = deepcopy(upgradeProperties[2017][key])
    upgradeProperties[2017][key+'PU']['ScenToRun'] = ['GenSim','DigiPU'] + \
                                                     (['RecoPU','HARVESTPU'] if '202' in key else ['RecoFakeHLTPU','HARVESTFakeHLTPU']) + \
                                                     (['Nano'] if 'Design' not in key else [])

upgradeProperties[2026] = {
    '2026D49' : {
        'Geom' : 'Extended2026D49',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T15',
        'Era' : 'Phase2C9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D60' : {
        'Geom' : 'Extended2026D60',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T15',
        'Era' : 'Phase2C10',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D64' : {
        'Geom' : 'Extended2026D64',                   # N.B.: Geometry with square 50x50 um2 pixels in the Inner Tracker.
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T22',
        'Era' : 'Phase2C11',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D65' : {
        'Geom' : 'Extended2026D65',                   # N.B.: Geometry with 3D pixels in the Inner Tracker.
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T23',           # This symbolic GT has no pixel template / GenError informations.
        'ProcessModifier': 'phase2_PixelCPEGeneric',  # This modifier removes all need for IT template information. DO NOT USE for standard planar sensors.
        'Era' : 'Phase2C11',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D66' : {
        'Geom' : 'Extended2026D66',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C11',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D67' : {
        'Geom' : 'Extended2026D67',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C11M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D68' : {
        'Geom' : 'Extended2026D68',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C11',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D69' : {
        'Geom' : 'Extended2026D69',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C12',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D70' : {
        'Geom' : 'Extended2026D70',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C11',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D71' : {
        'Geom' : 'Extended2026D71',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C11',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D72' : {
        'Geom' : 'Extended2026D72',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C11_etlV4',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D73' : {
        'Geom' : 'Extended2026D73',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C11_etlV4',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
    },
    '2026D74' : {
        'Geom' : 'Extended2026D74',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic_T21',
        'Era' : 'Phase2C11M9',
        'ScenToRun' : ['GenSimHLBeamSpot','DigiTrigger','RecoGlobal', 'HARVESTGlobal'],
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
    ('WM_14TeV_TuneCUETP8M1_cfi', UpgradeFragment(Kby(9,100),'WM_14TeV')),
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
    ('DisplacedSUSY_stopToBottom_M_800_500mm_TuneCP5_14TeV_pythia8_cff', UpgradeFragment(Kby(9,50),'DisplacedSUSY_stopToB_M_800_500mm_14')),
])
