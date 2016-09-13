import FWCore.ParameterSet.Config as cms

gmtStage2Raw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::GMTSetup"),
    InputLabel = cms.InputTag("simGmtStage2Digis"),
    BMTFInputLabel = cms.InputTag("simBmtfDigis", "BMTF"),
    OMTFInputLabel = cms.InputTag("simOmtfDigis", "OMTF"),
    EMTFInputLabel = cms.InputTag("simEmtfDigis", "EMTF"),
    ImdInputLabelBMTF = cms.InputTag("simGmtStage2Digis", "imdMuonsBMTF"),
    ImdInputLabelEMTFNeg = cms.InputTag("simGmtStage2Digis", "imdMuonsEMTFNeg"),
    ImdInputLabelEMTFPos = cms.InputTag("simGmtStage2Digis", "imdMuonsEMTFPos"),
    ImdInputLabelOMTFNeg = cms.InputTag("simGmtStage2Digis", "imdMuonsOMTFNeg"),
    ImdInputLabelOMTFPos = cms.InputTag("simGmtStage2Digis", "imdMuonsOMTFPos"),
    FedId = cms.int32(1402),
    FWId = cms.uint32(1),
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8)
)
