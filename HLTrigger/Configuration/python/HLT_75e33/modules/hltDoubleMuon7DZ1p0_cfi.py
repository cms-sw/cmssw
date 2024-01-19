import FWCore.ParameterSet.Config as cms

hltDoubleMuon7DZ1p0 = cms.EDFilter("HLT2L1P2GTCandL1P2GTCandDZ",
    MaxDZ = cms.double(1.0),
    MinDR = cms.double(-1),
    MinN = cms.int32(1),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    l1GTAlgoName1 = cms.string("pDoubleTkMuon15_7"),
    l1GTAlgoName2 = cms.string("pDoubleTkMuon15_7"),
    originTag1 = cms.VInputTag(cms.InputTag("l1tGTProducer", "GMTTkMuons")),
    originTag2 = cms.VInputTag(cms.InputTag("l1tGTProducer", "GMTTkMuons")),
    saveTags = cms.bool(True),
    triggerType1 = cms.int32(-114),
    triggerType2 = cms.int32(-114)
)
