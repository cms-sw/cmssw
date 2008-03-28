import FWCore.ParameterSet.Config as cms

l1thcaltpg = cms.EDFilter("L1THCALTPG",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    hcaltpgSource = cms.InputTag("hcalDigis","","DQM"),
    MonitorDaemon = cms.untracked.bool(True),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False)
)


