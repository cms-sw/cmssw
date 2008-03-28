import FWCore.ParameterSet.Config as cms

l1trpctpg = cms.EDFilter("L1TRPCTPG",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    rpctpgSource = cms.InputTag("rpcunpacker","","DQM"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)


