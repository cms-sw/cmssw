import FWCore.ParameterSet.Config as cms

gmtStage2Digis = cms.EDProducer(
    "L1TRawToDigi",
    Setup = cms.string("stage2::GMTSetup"),
    InputLabel = cms.InputTag("rawDataCollector"),
    FedIds = cms.vint32(1402),
    FWId = cms.uint32(1),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8),
    lenAMCHeader = cms.untracked.int32(8),
    lenAMCTrailer = cms.untracked.int32(0),
    lenAMC13Header = cms.untracked.int32(8),
    lenAMC13Trailer = cms.untracked.int32(8)
)
