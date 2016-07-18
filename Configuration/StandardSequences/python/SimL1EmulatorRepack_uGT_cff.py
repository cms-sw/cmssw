import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

## L1REPACK FULL:  Re-Emulate all of L1 and repack into RAW


if not (eras.stage2L1Trigger.isChosen()):
    print "L1T WARN:  L1REPACK:Full (intended for 2016 data) only supports Stage 2 eras for now."
    print "L1T WARN:  Use a legacy version of L1REPACK for now."
else:
    print "L1T INFO:  L1REPACK:uGT (intended for 2016 data) will unpack uGMT and CaloLaye2 outputs and re-emulate uGT"

    # First, inputs to uGT:
    import EventFilter.L1TRawToDigi.gmtStage2Digis_cfi
    unpackGmtStage2 = EventFilter.L1TRawToDigi.gmtStage2Digis_cfi.gmtStage2Digis.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))    

    import EventFilter.L1TRawToDigi.caloStage2Digis_cfi
    unpackCaloStage2 = EventFilter.L1TRawToDigi.caloStage2Digis_cfi.caloStage2Digis.clone(
        InputLabel = cms.InputTag( 'rawDataCollector', processName=cms.InputTag.skipCurrentProcess()))    

    from L1Trigger.Configuration.SimL1Emulator_cff import *
    simGtStage2Digis.MuonInputTag   = cms.InputTag("unpackGmtStage2","Muon")
    simGtStage2Digis.EGammaInputTag = cms.InputTag("unpackCaloStage2","EGamma")
    simGtStage2Digis.TauInputTag    = cms.InputTag("unpackCaloStage2","Tau")
    simGtStage2Digis.JetInputTag    = cms.InputTag("unpackCaloStage2","Jet")
    simGtStage2Digis.EtSumInputTag  = cms.InputTag("unpackCaloStage2","EtSum")
    simGtStage2Digis.ExtInputTag    = cms.InputTag("simGtExtFakeStage2Digis") # as in default


    # Finally, pack the new L1T output back into RAW
    
    # pack simulated uGT
    from EventFilter.L1TRawToDigi.gtStage2Raw_cfi import gtStage2Raw as packGtStage2
    packGtStage2.MuonInputTag   = cms.InputTag("unpackGmtStage2","Muon")
    packGtStage2.EGammaInputTag = cms.InputTag("unpackCaloStage2","EGamma")
    packGtStage2.TauInputTag    = cms.InputTag("unpackCaloStage2","Tau")
    packGtStage2.JetInputTag    = cms.InputTag("unpackCaloStage2","Jet")
    packGtStage2.EtSumInputTag  = cms.InputTag("unpackCaloStage2","EtSum")
    packGtStage2.GtInputTag     = cms.InputTag("simGtStage2Digis") # as in default
    packGtStage2.ExtInputTag    = cms.InputTag("simGtStage2Digis") # as in default
    

    # combine the new L1 RAW with existing RAW for other FEDs
    import EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi
    rawDataCollector = EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi.rawDataCollector.clone(
        verbose = cms.untracked.int32(0),
        RawCollectionList = cms.VInputTag(
            cms.InputTag('packGtStage2'),
            cms.InputTag('rawDataCollector', processName=cms.InputTag.skipCurrentProcess()),
            )
        )


    
    SimL1Emulator = cms.Sequence(unpackGmtStage2
                                +unpackCaloStage2
                                +SimL1TechnicalTriggers
                                +SimL1TGlobal
                                +packGtStage2
                                +rawDataCollector)
