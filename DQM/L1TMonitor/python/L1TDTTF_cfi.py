import FWCore.ParameterSet.Config as cms

l1tdttf = cms.EDFilter("L1TDTTF",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    dttfSource = cms.InputTag("l1GtUnpack","","DQM"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)


