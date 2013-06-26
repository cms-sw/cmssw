import FWCore.ParameterSet.Config as cms

l1tGt = cms.EDAnalyzer("L1TGT",
    gtSource = cms.InputTag("gtDigis"),    
    gtEvmSource = cms.InputTag("gtEvmDigis"),
    runInEventLoop=cms.untracked.bool(False),
    runInEndLumi=cms.untracked.bool(True),
    runInEndRun=cms.untracked.bool(True),
    runInEndJob=cms.untracked.bool(False),
    verbose = cms.untracked.bool(False)
)


