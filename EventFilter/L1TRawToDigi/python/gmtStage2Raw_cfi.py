import FWCore.ParameterSet.Config as cms

gmtStage2Raw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::GMTSetup"),
    InputLabel = cms.InputTag("simGmtStage2Digis"),
    BMTFInputLabel = cms.InputTag("simBmtfDigis", "BMTF"),
    OMTFInputLabel = cms.InputTag("simOmtfDigis", "OMTF"),
    EMTFInputLabel = cms.InputTag("simEmtfDigis", "EMTF"),
    FedId = cms.int32(1402),
    FWId = cms.uint32(1),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8)
)
