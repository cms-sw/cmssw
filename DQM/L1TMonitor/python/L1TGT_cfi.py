import FWCore.ParameterSet.Config as cms

l1tgt = cms.EDFilter("L1TGT",
    DQMStore = cms.untracked.bool(True),
    gtSource = cms.InputTag("l1GtUnpack","","DQM"),
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)


