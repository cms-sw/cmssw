import FWCore.ParameterSet.Config as cms

l1tgct = cms.EDAnalyzer("L1TGCT",
    verbose = cms.untracked.bool(False),
    gctCentralJetsSource = cms.InputTag("l1GctHwDigis","cenJets","DQM"),
    gctNonIsoEmSource = cms.InputTag("l1GctHwDigis","nonIsoEm","DQM"),
    DQMStore = cms.untracked.bool(True),
    gctForwardJetsSource = cms.InputTag("l1GctHwDigis","forJets","DQM"),
    gctIsoEmSource = cms.InputTag("l1GctHwDigis","isoEm","DQM"),
    gctEnergySumsSource = cms.InputTag("l1GctHwDigis","","DQM"),
    disableROOToutput = cms.untracked.bool(True),
    gctTauJetsSource = cms.InputTag("l1GctHwDigis","tauJets","DQM")
)


