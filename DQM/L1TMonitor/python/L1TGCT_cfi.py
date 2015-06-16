import FWCore.ParameterSet.Config as cms

l1tGct = cms.EDAnalyzer("L1TGCT",
    gctCentralJetsSource = cms.InputTag("gctDigis","cenJets"),
    gctForwardJetsSource = cms.InputTag("gctDigis","forJets"),
    gctTauJetsSource = cms.InputTag("gctDigis","tauJets"),
    gctIsoTauJetsSource = cms.InputTag("gctDigis","fake"),
    gctEnergySumsSource = cms.InputTag("gctDigis"),
    gctIsoEmSource = cms.InputTag("gctDigis","isoEm"),
    gctNonIsoEmSource = cms.InputTag("gctDigis","nonIsoEm"),
    monitorDir = cms.untracked.string("L1T/L1TGCT"),
    verbose = cms.untracked.bool(False),
    stage1_layer2_ = cms.bool(False),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    filterTriggerType = cms.int32(1)
)


