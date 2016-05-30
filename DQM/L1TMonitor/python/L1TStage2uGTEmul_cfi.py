import FWCore.ParameterSet.Config as cms

l1tStage2uGtEmul = cms.EDAnalyzer("L1TStage2uGT",
    l1tStage2uGtSource = cms.InputTag("valGtStage2Digis"),    
    monitorDir = cms.untracked.string("L1T2016EMU/L1TStage2uGTEmul"),
    verbose = cms.untracked.bool(False)
)
