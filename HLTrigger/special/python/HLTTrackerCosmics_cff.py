import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
level1seedHLTTrackerCosmics = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleHLTTrackerCosmics = copy.deepcopy(hltPrescaler)
hltTrackerCosmics = cms.Sequence(level1seedHLTTrackerCosmics+prescaleHLTTrackerCosmics)
level1seedHLTTrackerCosmics.L1TechTriggerSeeding = True
level1seedHLTTrackerCosmics.L1SeedsLogicalExpression = "0"
prescaleHLTTrackerCosmics.prescaleFactor = 1

