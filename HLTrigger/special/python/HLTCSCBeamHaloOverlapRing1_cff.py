import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
level1seedHLTCSCBeamHaloOverlapRing1 = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
prescaleHLTCSCBeamHaloOverlapRing1 = copy.deepcopy(hltPrescaler)
overlapsHLTCSCBeamHaloOverlapRing1 = cms.EDFilter("HLTCSCOverlapFilter",
    fillHists = cms.bool(False),
    minHits = cms.uint32(4),
    ring2 = cms.bool(False),
    ring1 = cms.bool(True),
    yWindow = cms.double(2.0),
    input = cms.InputTag("csc2DRecHits"),
    xWindow = cms.double(2.0)
)

hltCSCBeamHaloOverlapRing1 = cms.Sequence(level1seedHLTCSCBeamHaloOverlapRing1+prescaleHLTCSCBeamHaloOverlapRing1+cms.SequencePlaceholder("muonCSCDigis")*cms.SequencePlaceholder("csc2DRecHits")*overlapsHLTCSCBeamHaloOverlapRing1)
#replace level1seedHLTCSCBeamHaloOverlapRing1.L1SeedsLogicalExpression = "L1_SingleMuBeamHalo"
level1seedHLTCSCBeamHaloOverlapRing1.L1TechTriggerSeeding = True
level1seedHLTCSCBeamHaloOverlapRing1.L1SeedsLogicalExpression = 3
prescaleHLTCSCBeamHaloOverlapRing1.prescaleFactor = 1

