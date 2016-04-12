import FWCore.ParameterSet.Config as cms

l1tComparisonStage2RAWvsEMU = cms.EDAnalyzer(
    "L1TComparison",
    tag        = cms.string("Stage 2 Unpacked Digis"),
    egCheck    = cms.bool(True),
    tauCheck   = cms.bool(True),
    jetCheck   = cms.bool(True),
    sumCheck   = cms.bool(True),
    muonCheck  = cms.bool(True),
    algCheck   = cms.bool(False),
    bxZeroOnly = cms.bool(True),

    egTagA     = cms.InputTag("simCaloStage2Digis"),
    tauTagA    = cms.InputTag("simCaloStage2Digis"),
    jetTagA    = cms.InputTag("simCaloStage2Digis"),
    sumTagA    = cms.InputTag("simCaloStage2Digis"),
    muonTagA   =  cms.InputTag("simGmtStage2Digis",""),
    algTagA    =  cms.InputTag("simGtStage2Digis",""),

    # for initial module testing compared unpacked to unpacked!!!
    egTagB     = cms.InputTag("caloStage2Digis"),
    tauTagB    = cms.InputTag("caloStage2Digis"),
    jetTagB    = cms.InputTag("caloStage2Digis"),
    sumTagB    = cms.InputTag("caloStage2Digis"),
    muonTagB   =  cms.InputTag("gmtStage2Digis",""),
    algTagB    =  cms.InputTag("gtStage2Digis",""),
)
