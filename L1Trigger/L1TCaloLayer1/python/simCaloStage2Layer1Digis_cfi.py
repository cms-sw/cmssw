#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms

simCaloStage2Layer1Digis = cms.EDProducer(
    'L1TCaloLayer1',
    ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    hcalToken = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    useLSB = cms.bool(True),
    useCalib = cms.bool(True),
    useECALLUT = cms.bool(True),
    useHCALLUT = cms.bool(True),
    useHFLUT = cms.bool(True),
    useHCALFBLUT = cms.bool(False),
    verbose = cms.untracked.bool(False),
    unpackEcalMask = cms.bool(False),
    unpackHcalMask = cms.bool(False),
    # See UCTLayer1.hh for firmware version
    firmwareVersion = cms.int32(1),
    )

from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
stage2L1Trigger_2017.toModify( simCaloStage2Layer1Digis, firmwareVersion = cms.int32(3) )
