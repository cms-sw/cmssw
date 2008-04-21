import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
level1seedHLTForwardBSC = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleHLTForwardBSC = copy.deepcopy(hltPrescaler)
hltForwardBSC = cms.Sequence(level1seedHLTForwardBSC+prescaleHLTForwardBSC)
level1seedHLTForwardBSC.L1TechTriggerSeeding = True
level1seedHLTForwardBSC.L1SeedsLogicalExpression = '36 OR 37 OR 42'
prescaleHLTForwardBSC.prescaleFactor = 1

