import FWCore.ParameterSet.Config as cms

l1tStage1Layer2 = cms.EDAnalyzer("L1TGCT",
    gctCentralJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets"),
    gctForwardJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets"),
    gctTauJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets"),
    gctIsoTauJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","isoTauJets"),       
    gctEnergySumsSource = cms.InputTag("caloStage1LegacyFormatDigis"),
    gctIsoEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm"),
    gctNonIsoEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm"),
    monitorDir = cms.untracked.string("L1TStage1/L1TStage1Layer2"),
    stage1_layer2_ = cms.bool(True),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(True),
    filterTriggerType = cms.int32(1)
)

