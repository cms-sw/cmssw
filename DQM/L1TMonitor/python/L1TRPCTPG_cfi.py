import FWCore.ParameterSet.Config as cms

l1trpctpg = cms.EDFilter("L1TRPCTPG",
    disableROOToutput = cms.untracked.bool(True),
    rpctpgSource = cms.InputTag("rpcunpacker","","DQM"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


