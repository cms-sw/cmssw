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
    verbose = cms.bool(False),
    unpackEcalMask = cms.bool(False),
    unpackHcalMask = cms.bool(False),
    )
