import FWCore.ParameterSet.Config as cms

l1tecaltpg = cms.EDFilter("L1TECALTPG",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    ecaltpgSourceE = cms.InputTag("ecalEBunpacker","EETT","DQM"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    ecaltpgSourceB = cms.InputTag("ecalEBunpacker","EBTT","DQM"),
    disableROOToutput = cms.untracked.bool(True)
)


