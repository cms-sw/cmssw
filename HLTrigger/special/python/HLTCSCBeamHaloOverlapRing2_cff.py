import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
level1seedHLTCSCBeamHaloOverlapRing2 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleHLTCSCBeamHaloOverlapRing2 = copy.deepcopy(hltPrescaler)
overlapsHLTCSCBeamHaloOverlapRing2 = cms.EDFilter("HLTCSCOverlapFilter",
    fillHists = cms.bool(False),
    minHits = cms.uint32(4),
    ring2 = cms.bool(True),
    ring1 = cms.bool(False),
    yWindow = cms.double(2.0),
    input = cms.InputTag("csc2DRecHits"),
    xWindow = cms.double(2.0)
)

hltCSCBeamHaloOverlapRing2 = cms.Sequence(level1seedHLTCSCBeamHaloOverlapRing2+prescaleHLTCSCBeamHaloOverlapRing2+cms.SequencePlaceholder("muonCSCDigis")*cms.SequencePlaceholder("csc2DRecHits")*overlapsHLTCSCBeamHaloOverlapRing2)
#replace level1seedHLTCSCBeamHaloOverlapRing2.L1SeedsLogicalExpression = "L1_SingleMuBeamHalo"
level1seedHLTCSCBeamHaloOverlapRing2.L1TechTriggerSeeding = True
level1seedHLTCSCBeamHaloOverlapRing2.L1SeedsLogicalExpression = 3
prescaleHLTCSCBeamHaloOverlapRing2.prescaleFactor = 1

