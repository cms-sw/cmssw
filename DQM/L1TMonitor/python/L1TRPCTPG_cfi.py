import FWCore.ParameterSet.Config as cms

l1trpctpg = cms.EDFilter("L1TRPCTPG",
    disableROOToutput = cms.untracked.bool(True),
    rpctpgSource = cms.InputTag("rpcunpacker","","DQM"),
    rpctfSource = cms.InputTag("l1GtUnpack","","DQM"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


