import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
level1seedHLTCSCBeamHalo = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleHLTCSCBeamHalo = copy.deepcopy(hltPrescaler)
hltCSCBeamHalo = cms.Sequence(level1seedHLTCSCBeamHalo+prescaleHLTCSCBeamHalo)
level1seedHLTCSCBeamHalo.L1TechTriggerSeeding = True
level1seedHLTCSCBeamHalo.L1SeedsLogicalExpression = "3"
prescaleHLTCSCBeamHalo.prescaleFactor = 1

