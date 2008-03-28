import FWCore.ParameterSet.Config as cms

l1tdttpg = cms.EDFilter("L1TDTTPG",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    dttpgSource = cms.InputTag("l1tdttpgunpack","","DQM")
)


