import FWCore.ParameterSet.Config as cms

caloStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage2::CaloSetup"),
    InputLabel = cms.InputTag("rawDataCollector"),
    FedId = cms.int32(1301),
    FWId = cms.untracked.int32(2)
)
