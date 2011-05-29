import FWCore.ParameterSet.Config as cms

l1tGct = cms.EDAnalyzer("L1TGCT",
    gctCentralJetsSource = cms.InputTag("gctDigis","cenJets","DQM"),
    gctForwardJetsSource = cms.InputTag("gctDigis","forJets","DQM"),
    gctTauJetsSource = cms.InputTag("gctDigis","tauJets","DQM"),
    gctEnergySumsSource = cms.InputTag("gctDigis","","DQM"),
    gctIsoEmSource = cms.InputTag("gctDigis","isoEm","DQM"),
    gctNonIsoEmSource = cms.InputTag("gctDigis","nonIsoEm","DQM"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True)
)


