import FWCore.ParameterSet.Config as cms

l1tcsctf = cms.EDFilter("L1TCSCTF",
    disableROOToutput = cms.untracked.bool(True),
    csctfSource = cms.InputTag("l1GtUnpack","","DQM"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


