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
    FWId = cms.uint32(0x6000000), # FW version in GMT with displaced muon information
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8)
)

## Era: Run2_2016
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(gmtStage2Raw, BMTFInputLabel = cms.InputTag("simBmtfDigis", "BMTF"), FWId = cms.uint32(0x3000000))

## Era: Run2_2017
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
stage2L1Trigger_2017.toModify(gmtStage2Raw, BMTFInputLabel = cms.InputTag("simBmtfDigis", "BMTF"), FWId = cms.uint32(0x4010000))

### Era: Run2_2018
from Configuration.Eras.Modifier_stage2L1Trigger_2018_cff import stage2L1Trigger_2018
stage2L1Trigger_2018.toModify(gmtStage2Raw, BMTFInputLabel = cms.InputTag("simBmtfDigis", "BMTF"), FWId = cms.uint32(0x4010000))

### Era: Run3_2021
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(gmtStage2Raw, BMTFInputLabel = cms.InputTag("simKBmtfDigis", "BMTF"), FWId = cms.uint32(0x6000000))
