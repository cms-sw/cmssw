import FWCore.ParameterSet.Config as cms

caloStage1Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage1::CaloSetup"),
    #InputLabel = cms.InputTag("l1tDigiToRaw"),
    InputLabel = cms.InputTag("rawDataCollector"),
    FedId = cms.int32(1300),
    FWId = cms.untracked.int32(2),
    lenSlinkHeader = cms.untracked.int32(16),
    lenSlinkTrailer = cms.untracked.int32(8),
    lenAMCHeader = cms.untracked.int32(8),
    lenAMCTrailer = cms.untracked.int32(0),
    lenAMC13Header = cms.untracked.int32(8),
    lenAMC13Trailer = cms.untracked.int32(8)
)
