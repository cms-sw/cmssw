import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

## L1RE FULL:  Re-Emulate all of L1 


if not (eras.stage2L1Trigger.isChosen()):
    print "L1T WARN:  L1RE ALL only supports Stage 2 eras for now."
else:
    print "L1T INFO:  L1RE CALOuGT will unpack Calo Stage2 inputs, re-emulate Calo then uGT (Stage-2) using unpacked emulated uGMT."

    # First, Unpack all inputs to L1 Calo Layer-1:
    import EventFilter.EcalRawToDigi.EcalUnpackerData_cfi
    unpackEcal = EventFilter.EcalRawToDigi.EcalUnpackerData_cfi.ecalEBunpacker.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
    unpackHcal = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    # Second, Re-Emulate the entire L1T

    # Legacy trigger primitive emulations still running in 2016 trigger:
    # NOTE:  2016 HCAL HF TPs require a new emulation, which is not yet available...    
    from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
    simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('unpackHcal'),
        cms.InputTag('unpackHcal')
    )
    # not sure when/if this is needed...
    # HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)

    from L1Trigger.Configuration.SimL1Emulator_cff import *
    simCaloStage2Layer1Digis.ecalToken = cms.InputTag('unpackEcal:EcalTriggerPrimitives')
    simCaloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')
    # Picking up simulation a bit further downstream for now:

    SimL1Emulator = cms.Sequence(unpackEcal+unpackHcal
                                +simHcalTriggerPrimitiveDigis
                                +SimL1CaloAndGtEmulatorCore ### only Calo and uGT
                                )
