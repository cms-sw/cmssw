import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

def L1TCaloStage2ParamsForHW(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_HWConfig_cfi")
    return process

def L1TAddBitwiseLayer1(process):
    from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis as simCaloStage2BitwiseLayer1Digis
    from L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi import simCaloStage2Digis as simCaloStage2BitwiseDigis        
    process.simCaloStage2BitwiseLayer1Digis = simCaloStage2BitwiseLayer1Digis.clone()
    process.simCaloStage2BitwiseLayer1Digis.ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitives")
    process.simCaloStage2BitwiseDigis = simCaloStage2BitwiseDigis.clone()
    process.simCaloStage2BitwiseDigis.towerToken = cms.InputTag("simCaloStage2BitwiseLayer1Digis")
    process.SimL1TCalorimeter = cms.Sequence( process.simCaloStage2Layer1Digis + process.simCaloStage2Digis + process.simCaloStage2BitwiseLayer1Digis + process.simCaloStage2BitwiseDigis)    
    from L1Trigger.L1TNtuples.l1UpgradeTree_cfi import l1UpgradeTree
    process.l1UpgradeBitwiseTree = l1UpgradeTree.clone()
    process.l1UpgradeBitwiseTree.egToken = cms.untracked.InputTag("simCaloStage2BitwiseDigis")
    process.l1UpgradeBitwiseTree.tauTokens = cms.untracked.VInputTag("simCaloStage2BitwiseDigis")
    process.l1UpgradeBitwiseTree.jetToken = cms.untracked.InputTag("simCaloStage2BitwiseDigis")
    process.l1UpgradeBitwiseTree.muonToken = cms.untracked.InputTag("simGmtStage2Digis")
    process.l1UpgradeBitwiseTree.sumToken = cms.untracked.InputTag("simCaloStage2BitwiseDigis")
    process.l1ntuplebitwise = cms.Path(
        process.l1UpgradeBitwiseTree
    )
    process.schedule.append(process.l1ntuplebitwise)
    print "modified L1TReEmul:  "
    print process.L1TReEmul
    return process

# As of 80X, this ES configuration is needed for *data* GTs (mc tags work w/o)
def L1TEventSetupForHF1x1TPs(process):
    process.es_pool_hf1x1 = cms.ESSource(
        "PoolDBESSource",
        #process.CondDBSetup,
        timetype = cms.string('runnumber'),
        toGet = cms.VPSet(
            cms.PSet(record = cms.string("HcalLutMetadataRcd"),
                     tag = cms.string("HcalLutMetadata_HFTP_1x1")
                     ),
            cms.PSet(record = cms.string("HcalElectronicsMapRcd"),
                     tag = cms.string("HcalElectronicsMap_HFTP_1x1")
                     )
            ),
        connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
        authenticationMethod = cms.untracked.uint32(0)
        )
    process.es_prefer_es_pool_hf1x1 = cms.ESPrefer("PoolDBESSource", "es_pool_hf1x1")    
    return process

def L1TReEmulFromRAW2015(process):
    process.load('L1Trigger.Configuration.SimL1Emulator_cff')
    process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
    process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('hcalDigis'),
        cms.InputTag('hcalDigis')
    )
    process.L1TReEmul = cms.Sequence(process.simEcalTriggerPrimitiveDigis * process.simHcalTriggerPrimitiveDigis * process.SimL1Emulator)
    process.simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'  
    process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi')
    process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )  

    if eras.stage2L1Trigger.isChosen():
        process.simTwinMuxDigis.RPC_Source         = cms.InputTag('muonRPCDigis')
        # When available, this will switch to TwinMux input Digis:
        process.simTwinMuxDigis.DTDigi_Source      = cms.InputTag("dttfDigis")
        process.simTwinMuxDigis.DTThetaDigi_Source = cms.InputTag("dttfDigis")
        process.simOmtfDigis.srcRPC                = cms.InputTag('muonRPCDigis')
        process.simBmtfDigis.DTDigi_Source         = cms.InputTag("simTwinMuxDigis")
        process.simBmtfDigis.DTDigi_Theta_Source   = cms.InputTag("dttfDigis")
        process.simEmtfDigis.CSCInput              = cms.InputTag("csctfDigis")
        process.simEmtfDigis.RPCInput              = cms.InputTag('muonRPCDigis')
        process.simOmtfDigis.srcCSC                = cms.InputTag("csctfDigis")
        process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitives")
        process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
        process.schedule.append(process.L1TReEmulPath)
        print "L1TReEmul sequence:  "
        print process.L1TReEmul
        print process.schedule
        # quiet warning abouts missing Stage-2 payloads, since they won't reliably exist in 2015 data.
        if hasattr(process, "caloStage2Digis"):
            process.caloStage2Digis.MinFeds = cms.uint32(0)
        if hasattr(process, "gmtStage2Digis"):
            process.gmtStage2Digis.MinFeds = cms.uint32(0)
        if hasattr(process, "gtStage2Digis"):
            process.gtStage2Digis.MinFeds = cms.uint32(0)            
        return process
    else:
        process.simRctDigis.ecalDigis = cms.VInputTag('simEcalTriggerPrimitiveDigis')
        process.simRctDigis.hcalDigis = cms.VInputTag('simHcalTriggerPrimitiveDigis')
        process.simRpcTriggerDigis.label = 'muonRPCDigis'
        process.simRpcTechTrigDigis.RPCDigiLabel  = 'muonRPCDigis'
        process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
        process.schedule.append(process.L1TReEmulPath)
        print "L1TReEmul sequence:  "
        print process.L1TReEmul
        print process.schedule
        return process

def L1TReEmulMCFromRAW2015(process):
    L1TReEmulFromRAW2015(process)
    if eras.stage2L1Trigger.isChosen():
            process.simEmtfDigis.CSCInput           = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED')
            process.simOmtfDigis.srcCSC             = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED')
    return process

def L1TReEmulFromRAW2015simCaloTP(process):
    L1TReEmulFromRAW2015(process)
    if eras.stage2L1Trigger.isChosen():
            process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
    return process

def L1TReEmulFromRAW(process):
    process.load('L1Trigger.Configuration.SimL1Emulator_cff')
    process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
    process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('hcalDigis'),
        cms.InputTag('hcalDigis')
    )
    process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi')
    process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )  
    process.L1TReEmul = cms.Sequence(process.simEcalTriggerPrimitiveDigis * process.simHcalTriggerPrimitiveDigis * process.SimL1Emulator)
    if eras.stage2L1Trigger.isChosen():
        #cutlist=['simDtTriggerPrimitiveDigis','simCscTriggerPrimitiveDigis']
        #for b in cutlist:
        #    process.SimL1Emulator.remove(getattr(process,b))
        # TwinMux
        process.simTwinMuxDigis.RPC_Source         = cms.InputTag('muonRPCDigis')
        process.simTwinMuxDigis.DTDigi_Source      = cms.InputTag('bmtfDigis')
        process.simTwinMuxDigis.DTThetaDigi_Source = cms.InputTag('bmtfDigis')
        # BMTF
        process.simBmtfDigis.DTDigi_Source         = cms.InputTag('bmtfDigis')
        process.simBmtfDigis.DTDigi_Theta_Source   = cms.InputTag('bmtfDigis')
        # OMTF
        process.simOmtfDigis.srcRPC                = cms.InputTag('muonRPCDigis')
        process.simOmtfDigis.srcCSC                = cms.InputTag('csctfDigis')
        process.simOmtfDigis.srcDTPh               = cms.InputTag('bmtfDigis')
        process.simOmtfDigis.srcDTTh               = cms.InputTag('bmtfDigis')
        # EMTF
        process.simEmtfDigis.CSCInput              = cms.InputTag('emtfStage2Digis')
        process.simEmtfDigis.RPCInput              = cms.InputTag('muonRPCDigis')
        # Calo Layer1
        process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag('ecalDigis:EcalTriggerPrimitives')
        process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag('hcalDigis:')
        process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
        process.schedule.append(process.L1TReEmulPath)
        print "L1TReEmulPath sequence:  "
        print process.L1TReEmulPath
        print process.schedule
        return process
    else:
        process.simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:EcalTriggerPrimitives' ) )
        process.simRctDigis.hcalDigis = cms.VInputTag('hcalDigis:')
        process.simRpcTriggerDigis.label         = 'muonRPCDigis'
        process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
        process.schedule.append(process.L1TReEmulPath)
        print "L1TReEmul sequence:  "
        print process.L1TReEmul
        print process.schedule
        return process

def L1TReEmulMCFromRAW(process):
    L1TReEmulFromRAW(process)
    if eras.stage2L1Trigger.isChosen():
            process.simEmtfDigis.CSCInput           = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED')
            process.simOmtfDigis.srcCSC             = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED')
    return process

def L1TReEmulFromRAWsimEcalTP(process):
    L1TReEmulFromRAW(process)
    if eras.stage2L1Trigger.isChosen():
            process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
    return process

def L1TReEmulFromRAWsimHcalTP(process):
    L1TReEmulFromRAW(process)
    if eras.stage2L1Trigger.isChosen():
            process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')
    return process

def L1TReEmulFromRAWsimTP(process):
    L1TReEmulFromRAW(process)
    if eras.stage2L1Trigger.isChosen():
        # TwinMux
        process.simTwinMuxDigis.RPC_Source         = cms.InputTag('muonRPCDigis')
        process.simTwinMuxDigis.DTDigi_Source      = cms.InputTag('simDtTriggerPrimitiveDigis')
        process.simTwinMuxDigis.DTThetaDigi_Source = cms.InputTag('simDtTriggerPrimitiveDigis')
        # BMTF
        process.simBmtfDigis.DTDigi_Source         = cms.InputTag('simTwinMuxDigis')
        process.simBmtfDigis.DTDigi_Theta_Source   = cms.InputTag('simDtTriggerPrimitiveDigis')
        # OMTF
        process.simOmtfDigis.srcRPC                = cms.InputTag('muonRPCDigis')
        process.simOmtfDigis.srcCSC                = cms.InputTag('simCscTriggerPrimitiveDigis')
        process.simOmtfDigis.srcDTPh               = cms.InputTag('simDtTriggerPrimitiveDigis')
        process.simOmtfDigis.srcDTTh               = cms.InputTag('simDtTriggerPrimitiveDigis')
        # EMTF
        process.simEmtfDigis.CSCInput              = cms.InputTag('simCscTriggerPrimitiveDigis')
        process.simEmtfDigis.RPCInput              = cms.InputTag('muonRPCDigis')
        # Layer1
        process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
        process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')
    return process

