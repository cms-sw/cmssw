import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
level1seedHLTCSCBeamHaloRing2or3 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleHLTCSCBeamHaloRing2or3 = copy.deepcopy(hltPrescaler)
filter23HLTCSCBeamHaloRing2or3 = cms.EDFilter("HLTCSCRing2or3Filter",
    input = cms.InputTag("csc2DRecHits"),
    xWindow = cms.double(2.0),
    minHits = cms.uint32(4),
    yWindow = cms.double(2.0)
)

hltCSCBeamHaloRing2or3 = cms.Sequence(level1seedHLTCSCBeamHaloRing2or3+prescaleHLTCSCBeamHaloRing2or3+cms.SequencePlaceholder("muonCSCDigis")*cms.SequencePlaceholder("csc2DRecHits")*filter23HLTCSCBeamHaloRing2or3)
level1seedHLTCSCBeamHaloRing2or3.L1TechTriggerSeeding = True
level1seedHLTCSCBeamHaloRing2or3.L1SeedsLogicalExpression = "3"
prescaleHLTCSCBeamHaloRing2or3.prescaleFactor = 1

