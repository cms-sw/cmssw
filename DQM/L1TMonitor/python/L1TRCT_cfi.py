import FWCore.ParameterSet.Config as cms

l1tRct = cms.EDAnalyzer("L1TRCT",
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    rctSource = cms.InputTag("l1GctHwDigis","","DQM"),
    verbose = cms.untracked.bool(False)
)


