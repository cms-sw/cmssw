import FWCore.ParameterSet.Config as cms

l1trpctf = cms.EDFilter("L1TRPCTF",
    disableROOToutput = cms.untracked.bool(True),
    rpctfSource = cms.InputTag("l1GtUnpack","","DQM"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


