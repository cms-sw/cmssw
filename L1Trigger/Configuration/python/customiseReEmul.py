import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

def L1TReEmulFromRAW(process):
    # just assume stage 1 for initial testing...
    if eras.stage2L1Trigger.isChosen():
        L1TReEmulStage2FromRAW(process)
    else:
        L1TReEmulStage1FromRAW(process)
    return process

# NOT TESTED YET:
#def L1TReEmulHCALTPFromRAW(process):
#    process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
#    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
#    process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
#        cms.InputTag('hcalDigis'),
#        cms.InputTag('hcalDigis')
#    )
#    # not sure what this does... (inherited)
#    # over-ride the corresponding Calo inputs
#    process.HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)
#    if hasattr(process,"simCaloStage2Layer1Digis"):        
#        process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')
#    if hasattr(process,"simRctDigis"):        
#        process.simRctDigis.hcalDigis = cms.VInputTag('simHcalTriggerPrimitiveDigis')
#    process.L1TReEmul = cms.Sequence(simHcalTriggerPrimitiveDigis * SimL1Emulator)
#    return process



# common to legacy / stage-1 / stage-2
def L1TReEmulCommonFromRAW(process):
    process.load('L1Trigger.Configuration.SimL1Emulator_cff')
    process.L1TReEmul = cms.Sequence(process.SimL1Emulator)   
    process.simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'  
    process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi')
    process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )  
    return process

def L1TReEmulStage2FromRAW(process):
    L1TReEmulCommonFromRAW(process)
    process.simTwinMuxDigis.RPC_Source         = cms.InputTag('muonRPCDigis')
    # When available, this will switch to TwinMux input Digis:
    process.simTwinMuxDigis.DTDigi_Source      = cms.InputTag("dttfDigis")
    process.simTwinMuxDigis.DTThetaDigi_Source = cms.InputTag("dttfDigis")
    process.simOmtfDigis.srcRPC                = cms.InputTag('muonRPCDigis')
    process.simBmtfDigis.DTDigi_Source         = cms.InputTag("simTwinMuxDigis")
    process.simBmtfDigis.DTDigi_Theta_Source   = cms.InputTag("dttfDigis")
    process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitives")
    process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag("hcalDigis")
    process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
    process.schedule.append(process.L1TReEmulPath)
    print "L1TReEmul sequence:  "
    print process.L1TReEmul
    print process.schedule
    return process

def L1TReEmulStage1FromRAW(process):
    L1TReEmulCommonFromRAW(process)
    process.simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:EcalTriggerPrimitives' ) )
    process.simRctDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'hcalTriggerPrimitiveDigis' ) ) # or ?
    process.simRpcTriggerDigis.label         = 'muonRPCDigis'
    process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
    process.schedule.append(process.L1TReEmulPath)
    print "L1TReEmul sequence:  "
    print process.L1TReEmul
    print process.schedule
    return process





