import FWCore.ParameterSet.Config as cms

l1trpctf = cms.EDFilter("L1TRPCTF",
    disableROOToutput = cms.untracked.bool(True),
    rpctfSource = cms.InputTag("l1GtUnpack","","DQM"),
    rpctfRPCDigiSource = cms.InputTag("rpcunpacker","DQM"),
    rateUpdateTime = cms.int32(20),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


