import FWCore.ParameterSet.Config as cms

caloLayer1RawFed1354 = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::CaloLayer1Setup"),
    ecalDigis = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    hcalDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    caloRegions = cms.InputTag("simCaloStage2Layer1Digis"),
    FedId = cms.int32(1354),
    FWId = cms.uint32(0x12345678),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8),
    CTP7 = cms.untracked.bool(True),
)
caloLayer1RawFed1356 = caloLayer1RawFed1354.clone()
caloLayer1RawFed1356.FedId = 1356
caloLayer1RawFed1358 = caloLayer1RawFed1354.clone()
caloLayer1RawFed1358.FedId = 1358

caloLayer1Raw = cms.Task(caloLayer1RawFed1354, caloLayer1RawFed1356, caloLayer1RawFed1358)
