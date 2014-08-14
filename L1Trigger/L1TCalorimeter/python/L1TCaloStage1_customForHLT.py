# customization fragments to run L1Emulator with hltGetConfiguration
#

import FWCore.ParameterSet.Config as cms

##############################################################################

def customiseL1CaloAndGtEmulatorsFromRaw(process):
    # customization fragment to run calorimeter emulators (TPGs and L1 calorimeter emulators) 
    # and GT emulator starting from a RAW file assuming that "RawToDigi_cff" and "SimL1Emulator_cff" 
    # have already been loaded

    ## # run Calo TPGs on unpacked digis
    ## process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
    ## process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
    ## process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
    ##     cms.InputTag('hcalDigis'), 
    ##     cms.InputTag('hcalDigis')
    ## )
    
    # do not run muon emulators - instead, use unpacked GMT digis for GT input 
    # (GMT digis produced by same module as the GT digis, as GT and GMT have common unpacker)
    process.simRpcTechTrigDigis.RPCDigiLabel = 'muonRPCDigis'                                                                                                                                                                                           
    process.simGtDigis.GmtInputTag = 'gtDigis'                                                                                                                                                                                                          

    # RCT
    # HCAL input would be from hcalDigis if hack not needed
    from L1Trigger.Configuration.SimL1Emulator_cff import simRctDigis
    simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:EcalTriggerPrimitives' ) )
    simRctDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'simHcalTriggerPrimitiveDigis' ) )

    # stage 1 itself
    from L1Trigger.L1TCalorimeter.L1TCaloStage1_cff import rctUpgradeFormatDigis
    ## process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_cff')
    rctUpgradeFormatDigis.regionTag = cms.InputTag("simRctDigis")
    rctUpgradeFormatDigis.emTag = cms.InputTag("simRctDigis")

    # GT
    from L1Trigger.Configuration.SimL1Emulator_cff import simGtDigis
    simGtDigis.GmtInputTag = 'gtDigis'
    simGtDigis.GctInputTag = 'caloStage1LegacyFormatDigis'
    simGtDigis.TechnicalTriggersInputTags = cms.VInputTag( )

    # run Calo TPGs, L1 GCT, technical triggers, L1 GT
    SimL1Emulator = cms.Sequence(
        process.L1TRerunHCALTP_FromRAW +
        process.ecalDigis +
        process.simRctDigis +
        process.L1TCaloStage1 +
        process.simGtDigis )

    # replace the SimL1Emulator in all paths and sequences
    for iterable in process.sequences.itervalues():
        iterable.replace( process.SimL1Emulator, SimL1Emulator)
    for iterable in process.paths.itervalues():
        iterable.replace( process.SimL1Emulator, SimL1Emulator)
    for iterable in process.endpaths.itervalues():
        iterable.replace( process.SimL1Emulator, SimL1Emulator)
    process.SimL1Emulator = SimL1Emulator

    return process

##############################################################################
