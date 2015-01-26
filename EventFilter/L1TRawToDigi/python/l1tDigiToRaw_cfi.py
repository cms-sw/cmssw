import FWCore.ParameterSet.Config as cms

l1tDigiToRaw = cms.EDProducer(
    "L1TDigiToRaw",
    Setup = cms.string("stage2::CaloSetup"),
    InputLabel = cms.InputTag("caloStage2FinalDigis"),
    FedId = cms.int32(1),
    FWId = cms.uint32(1)
)

#
# Make changes for Run 2
#
from Configuration.StandardSequences.Eras import eras
eras.run2.toModify( l1tDigiToRaw, InputLabel = cms.InputTag("simCaloStage1FinalDigis", "") )
eras.run2.toModify( l1tDigiToRaw, TauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "rlxTaus") )
eras.run2.toModify( l1tDigiToRaw, IsoTauInputLabel = cms.InputTag("simCaloStage1FinalDigis", "isoTaus") )
eras.run2.toModify( l1tDigiToRaw, HFBitCountsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFBitCounts") )
eras.run2.toModify( l1tDigiToRaw, HFRingSumsInputLabel = cms.InputTag("simCaloStage1FinalDigis", "HFRingSums") )
