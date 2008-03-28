import FWCore.ParameterSet.Config as cms

l1tgct = cms.EDFilter("L1TGCT",
    outputFile = cms.untracked.string('./L1TDQM.root'),
    verbose = cms.untracked.bool(False),
    MonitorDaemon = cms.untracked.bool(True),
    gctNonIsoEmSource = cms.InputTag("l1GctHwDigis","nonIsoEm","DQM"),
    gctEnergySumsSource = cms.InputTag("l1GctHwDigis","","DQM"),
    gctForwardJetsSource = cms.InputTag("l1GctHwDigis","forJets","DQM"),
    DaqMonitorBEInterface = cms.untracked.bool(True),
    gctCentralJetsSource = cms.InputTag("l1GctHwDigis","cenJets","DQM"),
    disableROOToutput = cms.untracked.bool(True),
    gctIsoEmSource = cms.InputTag("l1GctHwDigis","isoEm","DQM"),
    gctTauJetsSource = cms.InputTag("l1GctHwDigis","tauJets","DQM")
)


