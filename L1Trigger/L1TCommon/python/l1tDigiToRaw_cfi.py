import FWCore.ParameterSet.Config as cms

l1tDigiToRaw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage1::CaloSetup"),
    InputLabel = cms.InputTag("caloStage2FinalDigis"),
    FedId = cms.int32(1352),
    FWId = cms.uint32(2),
    RegionInputLabel = cms.InputTag("caloStage1Digis", ""),
    EmCandInputLabel = cms.InputTag("caloStage1Digis", "")
)

#
# Make some changes if running with the Stage 1 trigger
#
from Configuration.StandardSequences.Eras import eras
eras.stage1L1Trigger.toModify( l1tDigiToRaw, InputLabel = cms.InputTag("simCaloStage1FinalDigis", "") )
eras.stage1L1Trigger.toModify( l1tDigiToRaw, TauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "rlxTaus") )
eras.stage1L1Trigger.toModify( l1tDigiToRaw, IsoTauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "isoTaus") )
eras.stage1L1Trigger.toModify( l1tDigiToRaw, HFBitCountsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFBitCounts") )
eras.stage1L1Trigger.toModify( l1tDigiToRaw, HFRingSumsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFRingSums") )
eras.stage1L1Trigger.toModify( l1tDigiToRaw, RegionInputLabel = cms.InputTag("simRctDigis", "") )
eras.stage1L1Trigger.toModify( l1tDigiToRaw, EMCandInputLabel = cms.InputTag("simRctDigis", "") )
