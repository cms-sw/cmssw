import FWCore.ParameterSet.Config as cms

l1tcsctpg = cms.EDFilter("L1TCSCTPG",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    csctpgSource = cms.InputTag("L1CSCTPGUnpack","","DQM"),
    disableROOToutput = cms.untracked.bool(True)
)


