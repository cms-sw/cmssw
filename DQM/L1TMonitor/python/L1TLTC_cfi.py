import FWCore.ParameterSet.Config as cms

l1tltc = cms.EDFilter("L1TLTC",
    disableROOToutput = cms.untracked.bool(True),
    outputFile = cms.untracked.string('./L1TDQM.root'),
    MonitorDaemon = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    DaqMonitorBEInterface = cms.untracked.bool(True)
)


