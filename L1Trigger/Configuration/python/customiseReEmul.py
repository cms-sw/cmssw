import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

def L1TReEmulFromRAW(process):
    process.load('L1Trigger.Configuration.SimL1Emulator_cff')
    process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
    process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('hcalDigis'),
        cms.InputTag('hcalDigis')
    )
    process.L1TReEmul = cms.Sequence(process.simHcalTriggerPrimitiveDigis * process.SimL1Emulator)
    process.simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'  
    process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi')
    process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )  

    if eras.stage2L1Trigger.isChosen():
        process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')
        process.simTwinMuxDigis.RPC_Source         = cms.InputTag('muonRPCDigis')
        # When available, this will switch to TwinMux input Digis:
        process.simTwinMuxDigis.DTDigi_Source      = cms.InputTag("dttfDigis")
        process.simTwinMuxDigis.DTThetaDigi_Source = cms.InputTag("dttfDigis")
        process.simOmtfDigis.srcRPC                = cms.InputTag('muonRPCDigis')
        process.simBmtfDigis.DTDigi_Source         = cms.InputTag("simTwinMuxDigis")
        process.simBmtfDigis.DTDigi_Theta_Source   = cms.InputTag("dttfDigis")
        process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitives")
        process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
        process.schedule.append(process.L1TReEmulPath)
        print "L1TReEmul sequence:  "
        print process.L1TReEmul
        print process.schedule
        return process
    else:
        process.simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:EcalTriggerPrimitives' ) )
        process.simRctDigis.hcalDigis = cms.VInputTag('simHcalTriggerPrimitiveDigis')
        process.simRpcTriggerDigis.label         = 'muonRPCDigis'
        process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
        process.schedule.append(process.L1TReEmulPath)
        print "L1TReEmul sequence:  "
        print process.L1TReEmul
        print process.schedule
        return process
