import FWCore.ParameterSet.Config as cms

gmtStage2Raw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::GMTSetup"),
    InputLabel = cms.InputTag("gmtStage2Digis"),
    BMTFInputLabel = cms.InputTag("gmtStage2Digis", "BMTF"),
    OMTFInputLabel = cms.InputTag("gmtStage2Digis", "OMTF"),
    EMTFInputLabel = cms.InputTag("gmtStage2Digis", "EMTF"),
    FedId = cms.int32(1402),
    FWId = cms.uint32(1),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8)
)
