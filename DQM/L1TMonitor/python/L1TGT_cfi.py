import FWCore.ParameterSet.Config as cms

l1tgt = cms.EDFilter("L1TGT",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    gtSource = cms.InputTag("l1GtUnpack","","DQM"),
    disableROOToutput = cms.untracked.bool(True)
)


