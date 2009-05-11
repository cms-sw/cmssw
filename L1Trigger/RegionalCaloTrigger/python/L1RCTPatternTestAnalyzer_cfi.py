import FWCore.ParameterSet.Config as cms

L1RCTPatternTestAnalyzer = cms.EDAnalyzer("L1RCTPatternTestAnalyzer",
    hcalDigisLabel = cms.InputTag("hcalTriggerPrimitiveDigis"),
    showEmCands = cms.untracked.bool(True),
    testName = cms.untracked.string("none"),
    limitTo64 =  cms.untracked.bool(False),
    ecalDigisLabel = cms.InputTag("ecalTriggerPrimitiveDigis"),
    rctDigisLabel = cms.InputTag("rctDigis"),
    showRegionSums = cms.untracked.bool(True)
)



