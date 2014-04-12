import FWCore.ParameterSet.Config as cms

l1tGct = cms.EDAnalyzer("L1TGCT",
    gctCentralJetsSource = cms.InputTag("gctDigis","cenJets"),
    gctForwardJetsSource = cms.InputTag("gctDigis","forJets"),
    gctTauJetsSource = cms.InputTag("gctDigis","tauJets"),
    gctEnergySumsSource = cms.InputTag("gctDigis"),
    gctIsoEmSource = cms.InputTag("gctDigis","isoEm"),
    gctNonIsoEmSource = cms.InputTag("gctDigis","nonIsoEm"),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    filterTriggerType = cms.int32(1)
)


