import FWCore.ParameterSet.Config as cms

caloStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage2::CaloSetup"),
    InputLabel = cms.InputTag("rawDataCollector"),
    FedIds = cms.vint32( 1360, 1361 ),
    FWId = cms.untracked.int32(2)
)
