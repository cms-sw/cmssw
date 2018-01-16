import FWCore.ParameterSet.Config as cms

l1tStage2uGtEmul = DQMStep1Module('L1TStage2uGT',
    l1tStage2uGtSource = cms.InputTag("valGtStage2Digis"),    
    monitorDir = cms.untracked.string("L1TEMU/L1TStage2uGTEmul"),
    verbose = cms.untracked.bool(False)
)
