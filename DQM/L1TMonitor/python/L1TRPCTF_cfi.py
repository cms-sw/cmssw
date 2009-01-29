import FWCore.ParameterSet.Config as cms

l1trpctf = cms.EDFilter("L1TRPCTF",
    disableROOToutput = cms.untracked.bool(True),
    rpctfSource = cms.InputTag("l1GtUnpack","","DQM"),
    rpctfRPCDigiSource = cms.InputTag("rpcunpacker","DQM"),
    output_dir = cms.untracked.string('L1T/L1TRPCTF'),
    rateUpdateTime = cms.int32(20), # update after 20 seconds
    maxRateHistoSize = cms.int32(72000), # 20hours
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


