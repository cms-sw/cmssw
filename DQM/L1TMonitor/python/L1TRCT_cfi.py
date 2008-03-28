import FWCore.ParameterSet.Config as cms

l1trct = cms.EDFilter("L1TRCT",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    rctSource = cms.InputTag("l1GctHwDigis","","DQM")
)


