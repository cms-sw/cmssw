import FWCore.ParameterSet.Config as cms

l1tStage1Gt = cms.EDAnalyzer("L1TGT",
    gtSource = cms.InputTag("gtStage1Digis"),    
    gtEvmSource = cms.InputTag("gtStage1EvmDigis"),
    runInEventLoop=cms.untracked.bool(False),
    runInEndLumi=cms.untracked.bool(True),
    runInEndRun=cms.untracked.bool(True),
    runInEndJob=cms.untracked.bool(False),
    verbose = cms.untracked.bool(False),
    DirName = cms.untracked.string("L1T/L1TStage1Gt")
)
