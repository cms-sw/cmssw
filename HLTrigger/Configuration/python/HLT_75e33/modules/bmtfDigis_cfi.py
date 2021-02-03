import FWCore.ParameterSet.Config as cms

bmtfDigis = cms.EDProducer("L1TRawToDigi",
    FWId = cms.uint32(1),
    FedIds = cms.vint32(1376, 1377),
    InputLabel = cms.InputTag("rawDataCollector"),
    Setup = cms.string('stage2::BMTFSetup'),
    lenAMC13Header = cms.untracked.int32(8),
    lenAMC13Trailer = cms.untracked.int32(8),
    lenAMCHeader = cms.untracked.int32(8),
    lenAMCTrailer = cms.untracked.int32(0),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8)
)
