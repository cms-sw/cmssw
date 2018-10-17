from __future__ import print_function
import FWCore.ParameterSet.Config as cms

## L1REPACK FULL:  Re-Emulate all of L1 and repack into RAW


from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017

if not (stage2L1Trigger.isChosen()):
    print("L1T WARN:  L1REPACK:Full (intended for 2016 data) only supports Stage 2 eras for now.")
    print("L1T WARN:  Use a legacy version of L1REPACK for now.")
else:
    print("L1T INFO:  L1REPACK:Full (intended for 2016 & 2017 data) will unpack all L1T inputs, re-emulated (Stage-2), and pack uGT, uGMT, and Calo Stage-2 output.")

    # First, Unpack all inputs to L1:
    
    import EventFilter.L1TRawToDigi.bmtfDigis_cfi
    unpackBmtf = EventFilter.L1TRawToDigi.bmtfDigis_cfi.bmtfDigis.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))    

    import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
    unpackDttf = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone(
        DTTF_FED_Source = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess())) 
        
    import EventFilter.L1TRawToDigi.omtfStage2Digis_cfi
    unpackOmtf = EventFilter.L1TRawToDigi.omtfStage2Digis_cfi.omtfStage2Digis.clone(
        inputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))    
        
    import EventFilter.L1TRawToDigi.emtfStage2Digis_cfi
    unpackEmtf = EventFilter.L1TRawToDigi.emtfStage2Digis_cfi.emtfStage2Digis.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))    
        
    import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
    unpackCsctf = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone(
        producer = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))    

    import EventFilter.CSCRawToDigi.cscUnpacker_cfi
    unpackCSC = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone(
        InputObjects = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    import EventFilter.DTRawToDigi.dtunpacker_cfi
    unpackDT = EventFilter.DTRawToDigi.dtunpacker_cfi.muonDTDigis.clone(
        inputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
    unpackRPC = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    import EventFilter.RPCRawToDigi.rpcTwinMuxRawToDigi_cfi
    unpackRPCTwinMux = EventFilter.RPCRawToDigi.rpcTwinMuxRawToDigi_cfi.rpcTwinMuxRawToDigi.clone(
        inputTag = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    import EventFilter.L1TXRawToDigi.twinMuxStage2Digis_cfi
    unpackTwinMux = EventFilter.L1TXRawToDigi.twinMuxStage2Digis_cfi.twinMuxStage2Digis.clone(
        DTTM7_FED_Source = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))    
        
    import EventFilter.EcalRawToDigi.EcalUnpackerData_cfi
    unpackEcal = EventFilter.EcalRawToDigi.EcalUnpackerData_cfi.ecalEBunpacker.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
    unpackHcal = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    import EventFilter.L1TXRawToDigi.caloLayer1Stage2Digis_cfi
    unpackLayer1 = EventFilter.L1TXRawToDigi.caloLayer1Stage2Digis_cfi.l1tCaloLayer1Digis.clone(
        fedRawDataLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))

    # Second, Re-Emulate the entire L1T

    from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
    simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('unpackHcal'),
        cms.InputTag('unpackHcal')
    )
    simHcalTriggerPrimitiveDigis.inputUpgradeLabel = cms.VInputTag(
        cms.InputTag('unpackHcal'),     # upgrade HBHE
        cms.InputTag('unpackHcal')      # upgrade HF
    )

    from L1Trigger.Configuration.SimL1Emulator_cff import *
    
    simDtTriggerPrimitiveDigis.digiTag = 'unpackDT'
    simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'unpackCSC', 'MuonCSCComparatorDigi' )
    simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'unpackCSC', 'MuonCSCWireDigi' )

    simTwinMuxDigis.RPC_Source         = cms.InputTag('unpackRPCTwinMux')
    simTwinMuxDigis.DTDigi_Source      = cms.InputTag("unpackTwinMux:PhIn")
    simTwinMuxDigis.DTThetaDigi_Source = cms.InputTag("unpackTwinMux:ThIn")

    # -----------------------------------------------------------
    # change when availalbe simTwinMux and reliable DTTPs, CSCTPs
    cutlist=['simDtTriggerPrimitiveDigis','simCscTriggerPrimitiveDigis']
    for b in cutlist:
        SimL1EmulatorCore.remove(b)
    # -----------------------------------------------------------

    # BMTF
    simBmtfDigis.DTDigi_Source       = cms.InputTag("simTwinMuxDigis")
    simBmtfDigis.DTDigi_Theta_Source = cms.InputTag("unpackBmtf")

    # OMTF
    simOmtfDigis.srcRPC              = cms.InputTag('unpackRPC')
    simOmtfDigis.srcDTPh             = cms.InputTag("unpackBmtf")
    simOmtfDigis.srcDTTh             = cms.InputTag("unpackBmtf")
    simOmtfDigis.srcCSC              = cms.InputTag("unpackCsctf") 
    if (stage2L1Trigger_2017.isChosen()):
        simOmtfDigis.srcRPC          = cms.InputTag('unpackOmtf')
        simOmtfDigis.srcCSC          = cms.InputTag('unpackOmtf')
        simOmtfDigis.srcDTPh         = cms.InputTag('unpackOmtf')
        simOmtfDigis.srcDTTh         = cms.InputTag('unpackOmtf')

    # EMTF
    simEmtfDigis.CSCInput            = cms.InputTag("unpackEmtf") 
    simEmtfDigis.RPCInput            = cms.InputTag('unpackRPC')

    simCaloStage2Layer1Digis.ecalToken = cms.InputTag('unpackEcal:EcalTriggerPrimitives')
    simCaloStage2Layer1Digis.hcalToken = cms.InputTag('unpackLayer1')

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


    
    SimL1Emulator = cms.Sequence(unpackEcal+unpackHcal+unpackCSC+unpackDT+unpackRPC+unpackRPCTwinMux+unpackTwinMux+unpackOmtf+unpackEmtf+unpackCsctf+unpackBmtf
                                 +unpackLayer1
                                 +SimL1EmulatorCore+packCaloStage2
                                 +packGmtStage2+packGtStage2+rawDataCollector)

