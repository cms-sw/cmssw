import FWCore.ParameterSet.Config as cms

l1trpctf = cms.EDFilter("L1TRPCTF",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    rpctfSource = cms.InputTag("l1GtUnpack","","DQM"),
    MonitorDaemon = cms.untracked.bool(True),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)


