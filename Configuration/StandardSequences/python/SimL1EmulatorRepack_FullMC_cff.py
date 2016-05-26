import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

## L1REPACK FULL:  Re-Emulate all of L1 and repack into RAW


if not (eras.stage2L1Trigger.isChosen()):
    print "L1T WARN:  L1REPACK:FullMC (intended for MC events with RAW eventcontent) only supports Stage 2 eras for now."
    print "L1T WARN:  Use a legacy version of L1REPACK for now."
else:
    print "L1T INFO:  L1REPACK:FullMC  will unpack Calorimetry and Muon L1T inputs, re-emulate L1T (Stage-2), and pack uGT, uGMT, and Calo Stage-2 output."

    # First, Unpack all inputs to L1:

    import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
    unpackRPC = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    import EventFilter.DTRawToDigi.dtunpacker_cfi
    unpackDT = EventFilter.DTRawToDigi.dtunpacker_cfi.muonDTDigis.clone(
        inputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    import EventFilter.CSCRawToDigi.cscUnpacker_cfi
    unpackCSC = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone(
        InputObjects = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    import EventFilter.EcalRawToDigi.EcalUnpackerData_cfi
    unpackEcal = EventFilter.EcalRawToDigi.EcalUnpackerData_cfi.ecalEBunpacker.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
    unpackHcal = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    # Second, Re-Emulate the entire L1T
    #
    # Legacy trigger primitive emulations still running in 2016 trigger:
    #
    from SimCalorimetry.Configuration.SimCalorimetry_cff import *

    # Ecal TPs
    # cannot simulate EcalTPs, don't have EcalUnsuppressedDigis in RAW
    #     simEcalTriggerPrimitiveDigis.Label = 'unpackEcal'
    # further downstream, use unpacked EcalTPs

    # Hcal TPs
    simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('unpackHcal'),
        cms.InputTag('unpackHcal')
    )

    from L1Trigger.Configuration.SimL1Emulator_cff import *
    # DT TPs
    simDtTriggerPrimitiveDigis.digiTag                    = 'unpackDT'
    # CSC TPs
    simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag('unpackCSC', 'MuonCSCComparatorDigi' )
    simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'unpackCSC', 'MuonCSCWireDigi' )

    # TWIN-MUX
    simTwinMuxDigis.RPC_Source         = cms.InputTag('unpackRPC')
    simTwinMuxDigis.DTDigi_Source      = cms.InputTag("simDtTriggerPrimitiveDigis")
    simTwinMuxDigis.DTThetaDigi_Source = cms.InputTag("simDtTriggerPrimitiveDigis")

    # BMTF
    simBmtfDigis.DTDigi_Source       = cms.InputTag("simTwinMuxDigis")
    simBmtfDigis.DTDigi_Theta_Source = cms.InputTag("simDtTriggerPrimitiveDigis")

    # OMTF
    simOmtfDigis.srcRPC              = cms.InputTag('unpackRPC')
    simOmtfDigis.srcDTPh             = cms.InputTag("simDtTriggerPrimitiveDigis")
    simOmtfDigis.srcDTTh             = cms.InputTag("simDtTriggerPrimitiveDigis")
    simOmtfDigis.srcCSC              = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED')

    # EMTF
    simEmtfDigis.CSCInput            = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED')

    # CALO Layer1
    simCaloStage2Layer1Digis.ecalToken = cms.InputTag('unpackEcal:EcalTriggerPrimitives')
    simCaloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')

    # Finally, pack the new L1T output back into RAW
    from EventFilter.L1TRawToDigi.caloStage2Raw_cfi import caloStage2Raw as packCaloStage2
    from EventFilter.L1TRawToDigi.gmtStage2Raw_cfi import gmtStage2Raw as packGmtStage2
    from EventFilter.L1TRawToDigi.gtStage2Raw_cfi import gtStage2Raw as packGtStage2

    # combine the new L1 RAW with existing RAW for other FEDs
    import EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi
    rawDataCollector = EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi.rawDataCollector.clone(
        verbose = cms.untracked.int32(0),
        RawCollectionList = cms.VInputTag(
            cms.InputTag('packCaloStage2'),
            cms.InputTag('packGmtStage2'),
            cms.InputTag('packGtStage2'),
            cms.InputTag('rawDataCollector', processName=cms.InputTag.skipCurrentProcess()),
            )
        )

    SimL1Emulator = cms.Sequence(unpackRPC
                                + unpackDT
                                + unpackCSC
                                + unpackEcal
                                + unpackHcal
                                #+ simEcalTriggerPrimitiveDigis
                                + simHcalTriggerPrimitiveDigis
                                + SimL1EmulatorCore
                                + packCaloStage2
                                + packGmtStage2
                                + packGtStage2
                                + rawDataCollector)
