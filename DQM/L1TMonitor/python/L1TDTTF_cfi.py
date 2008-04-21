import FWCore.ParameterSet.Config as cms

l1tdttf = cms.EDFilter("L1TDTTF",
    disableROOToutput = cms.untracked.bool(True),
    dttfSource = cms.InputTag("l1GtUnpack","","DQM"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


