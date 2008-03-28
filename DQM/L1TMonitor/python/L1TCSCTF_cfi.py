import FWCore.ParameterSet.Config as cms

l1tcsctf = cms.EDFilter("L1TCSCTF",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    csctfSource = cms.InputTag("l1GtUnpack","","DQM"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)


