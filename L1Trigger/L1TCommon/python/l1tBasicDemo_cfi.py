import FWCore.ParameterSet.Config as cms

l1tBasicDemo = cms.EDAnalyzer(
    "L1TBasicDemo",
    UseTriggerBxOnly = cms.bool(True),
    EgTag            = cms.InputTag("caloStage2Digis"),
    TauTag           = cms.InputTag("caloStage2Digis"),
    JetTag           = cms.InputTag("caloStage2Digis"),
    SumTag           = cms.InputTag("caloStage2Digis"),
    MuonTag          = cms.InputTag("gmtStage2Digis", ""),
)


