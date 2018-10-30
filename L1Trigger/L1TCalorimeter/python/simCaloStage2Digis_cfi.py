import FWCore.ParameterSet.Config as cms

simCaloStage2Digis = cms.EDProducer(
    "L1TStage2Layer2Producer",
    towerToken = cms.InputTag("simCaloStage2Layer1Digis"),
    firmware = cms.int32(1),
    useStaticConfig = cms.bool(False)
)
