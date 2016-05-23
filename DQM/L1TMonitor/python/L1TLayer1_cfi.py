import FWCore.ParameterSet.Config as cms

l1tLayer1 = cms.EDAnalyzer("L1TLayer1",
    ecalTPSourceRecd = cms.InputTag("l1tCaloLayer1Digis"),
    hcalTPSourceRecd = cms.InputTag("l1tCaloLayer1Digis"),
    hfTPSourceRecd = cms.InputTag("l1tCaloLayer1Digis", "hfTPGDigis"),
    ecalTPSourceSent = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    hcalTPSourceSent = cms.InputTag("hcalDigis"),
    histFolder = cms.string('L1T2016/L1TLayer1'),
)
