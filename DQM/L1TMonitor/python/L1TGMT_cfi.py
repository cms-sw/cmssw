import FWCore.ParameterSet.Config as cms

l1tgmt = cms.EDAnalyzer("L1TGMT",
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    gmtSource = cms.InputTag("l1GtUnpack","","DQM"),
    DQMStore = cms.untracked.bool(True)
)


