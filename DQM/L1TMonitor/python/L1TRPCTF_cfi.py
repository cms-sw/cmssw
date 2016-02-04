import FWCore.ParameterSet.Config as cms

l1trpctf = cms.EDAnalyzer("L1TRPCTF",
    disableROOToutput = cms.untracked.bool(True),
    rpctfSource = cms.InputTag("l1GtUnpack","","DQM"),
    rpctfRPCDigiSource = cms.InputTag("rpcunpacker","DQM"),
    output_dir = cms.untracked.string('L1T/L1TRPCTF'),
    rateUpdateTime = cms.int32(20), # update after 20 seconds
    rateBinSize = cms.int32(60), # in seconds
    rateNoOfBins = cms.int32(3000),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True)
)


