import FWCore.ParameterSet.Config as cms

L1RCTTestAnalyzer = cms.EDAnalyzer("L1RCTTestAnalyzer",
    hcalDigisLabel = cms.InputTag("hcalTriggerPrimitiveDigis"),
    showEmCands = cms.untracked.bool(True),
    ecalDigisLabel = cms.InputTag("ecalTriggerPrimitiveDigis"),
    rctDigisLabel = cms.InputTag("rctDigis"),
    showRegionSums = cms.untracked.bool(True)
)



