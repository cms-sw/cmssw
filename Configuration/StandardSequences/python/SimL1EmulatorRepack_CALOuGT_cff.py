import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

## L1REPACK FULL:  Re-Emulate all of L1 and repack into RAW


if not (eras.stage2L1Trigger.isChosen()):
    print "L1T WARN:  L1REPACK ALL only supports Stage 2 eras for now."
    print "L1T WARN:  Use a legacy version of L1REPACK for now."
else:
    print "L1T INFO:  L1REPACK CALOuGT will unpack Calo Stage2 inputs, re-emulate Calo then uGT (Stage-2), and then pack uGT and Calo Stage-2 output."

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

    # Finally, pack the new L1T output back into RAW
    from EventFilter.L1TRawToDigi.caloStage2Raw_cfi import caloStage2Raw as packCaloStage2
    from EventFilter.L1TRawToDigi.gtStage2Raw_cfi import gtStage2Raw as packGtStage2

    # combine the new L1 RAW with existing RAW for other FEDs
    import EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi
    rawDataCollector = EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi.rawDataCollector.clone(
        verbose = cms.untracked.int32(0),
        RawCollectionList = cms.VInputTag(
            cms.InputTag('packCaloStage2'),
            cms.InputTag('packGtStage2'),
            cms.InputTag('rawDataCollector', processName=cms.InputTag.skipCurrentProcess()),
            )
        )


    
    SimL1Emulator = cms.Sequence(unpackEcal+unpackHcal
                                +simHcalTriggerPrimitiveDigis
                                +SimL1CaloAndGtEmulatorCore ### only Calo and uGT
                                +packCaloStage2
                                +packGtStage2
                                +rawDataCollector)
