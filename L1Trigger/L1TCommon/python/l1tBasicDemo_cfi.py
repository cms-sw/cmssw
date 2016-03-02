import FWCore.ParameterSet.Config as cms

l1tBasicDemo = cms.EDAnalyzer(
    "L1TBasicDemo",
    UseTriggerBxOnly = cms.bool(True),
    EgTag            = cms.InputTag("caloStage2Digis","EGamma"),
    TauTag           = cms.InputTag("caloStage2Digis","Tau"),
    JetTag           = cms.InputTag("caloStage2Digis","Jet"),
    SumTag           = cms.InputTag("caloStage2Digis","EtSum"),
    MuonTag          = cms.InputTag("gmtStage2Digis", "Muon"),
)


