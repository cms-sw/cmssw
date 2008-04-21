import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
level1seedHLTBackwardBSC = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleHLTBackwardBSC = copy.deepcopy(hltPrescaler)
hltBackwardBSC = cms.Sequence(level1seedHLTBackwardBSC+prescaleHLTBackwardBSC)
level1seedHLTBackwardBSC.L1TechTriggerSeeding = True
level1seedHLTBackwardBSC.L1SeedsLogicalExpression = '38 OR 39 OR 43'
prescaleHLTBackwardBSC.prescaleFactor = 1

