import FWCore.ParameterSet.Config as cms

gtStage2Raw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::GTSetup"),
#    TowerInputLabel = cms.InputTag("simCaloStage2Digis"),
    GtInputTag = cms.InputTag("simGtStage2Digis"),
    ExtInputTag = cms.InputTag("simGtExtFakeStage2Digis"),
    MuonInputTag   = cms.InputTag("simGmtStage2Digis"),
    EGammaInputTag = cms.InputTag("simCaloStage2Digis"),
    TauInputTag    = cms.InputTag("simCaloStage2Digis"),
    JetInputTag    = cms.InputTag("simCaloStage2Digis"),
    EtSumInputTag  = cms.InputTag("simCaloStage2Digis"),
    FedId = cms.int32(1404),
    FWId = cms.uint32(0x1120),  # FW w/ displaced muon info.
    lenSlinkHeader = cms.untracked.int32(8),
    lenSlinkTrailer = cms.untracked.int32(8)
)

## Era: Run2_2016
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(gtStage2Raw, FWId = cms.uint32(0x1000))  # FW w/o coordinates at vtx.

## Era: Run2_2017
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
stage2L1Trigger_2017.toModify(gtStage2Raw, FWId = cms.uint32(0x10A6)) # FW w/ vtx extrapolation.

### Era: Run2_2018
from Configuration.Eras.Modifier_stage2L1Trigger_2018_cff import stage2L1Trigger_2018
stage2L1Trigger_2018.toModify(gtStage2Raw, FWId = cms.uint32(0x10F2)) # FW w/ new HI centrality variables & vtx extrapolation.

### Era: Run3_2021
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(gtStage2Raw, FWId = cms.uint32(0x1120)) # FW w/ displaced muon info.

