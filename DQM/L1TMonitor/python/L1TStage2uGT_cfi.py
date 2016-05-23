import FWCore.ParameterSet.Config as cms

l1tStage2uGt = cms.EDAnalyzer("L1TStage2uGT",
    l1tStage2uGtSource = cms.InputTag("gtStage2Digis"),    
    verbose = cms.untracked.bool(False)
)
