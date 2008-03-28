import FWCore.ParameterSet.Config as cms

l1tcompare = cms.EDFilter("L1TCompare",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    ecalTpgSource = cms.InputTag("ecalTriggerPrimitiveDigis"),
    verbose = cms.untracked.bool(True),
    MonitorDaemon = cms.untracked.bool(True),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    gctSource = cms.InputTag("l1GctHwDigis","","DQM"),
    rctSource = cms.InputTag("l1GctHwDigis","","DQM")
)


