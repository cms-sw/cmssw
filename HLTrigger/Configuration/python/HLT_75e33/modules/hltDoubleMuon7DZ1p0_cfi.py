import FWCore.ParameterSet.Config as cms

hltDoubleMuon7DZ1p0 = cms.EDFilter("HLT2L1P2GTCandL1P2GTCandDZ",
    MaxDZ = cms.double(1.0),
    MinDR = cms.double(-1),
    MinN = cms.int32(1),
    algoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    algoName1 = cms.string("pDoubleTkMuon15_7"),
    algoName2 = cms.string("pDoubleTkMuon15_7"),
    originTag1 = cms.VInputTag(cms.InputTag("l1tGTProducer", "GMTTkMuons")),
    originTag2 = cms.VInputTag(cms.InputTag("l1tGTProducer", "GMTTkMuons")),
    saveTags = cms.bool(True),
    triggerType1 = cms.int32(-114),
    triggerType2 = cms.int32(-114)
)
