import FWCore.ParameterSet.Config as cms

l1tgmt = cms.EDFilter("L1TGMT",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    gmtSource = cms.InputTag("l1GtUnpack","","DQM"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)


