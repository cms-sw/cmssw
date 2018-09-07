from __future__ import print_function

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017

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
    print("# modified L1TReEmul:  ")
    print("# {0}".format(process.L1TReEmul))
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

    if stage2L1Trigger.isChosen():
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
        # quiet warning abouts missing Stage-2 payloads, since they won't reliably exist in 2015 data.
        if hasattr(process, "caloStage2Digis"):
            process.caloStage2Digis.MinFeds = cms.uint32(0)
        if hasattr(process, "gmtStage2Digis"):
            process.gmtStage2Digis.MinFeds = cms.uint32(0)
        if hasattr(process, "gtStage2Digis"):
            process.gtStage2Digis.MinFeds = cms.uint32(0)            
    else:
        process.simRctDigis.ecalDigis = cms.VInputTag('simEcalTriggerPrimitiveDigis')
        process.simRctDigis.hcalDigis = cms.VInputTag('simHcalTriggerPrimitiveDigis')
        process.simRpcTriggerDigis.label = 'muonRPCDigis'
        process.simRpcTechTrigDigis.RPCDigiLabel  = 'muonRPCDigis'
        process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
        process.schedule.append(process.L1TReEmulPath)

    print("# L1TReEmul sequence:  ")
    print("# {0}".format(process.L1TReEmul))
    print("# {0}".format(process.schedule))
    return process

def L1TReEmulMCFromRAW2015(process):
    L1TReEmulFromRAW2015(process)
    if stage2L1Trigger.isChosen():
            process.simEmtfDigis.CSCInput           = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED')
            process.simOmtfDigis.srcCSC             = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED')
    return process

def L1TReEmulFromRAW2015simCaloTP(process):
    L1TReEmulFromRAW2015(process)
    if stage2L1Trigger.isChosen():
            process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
    return process

def L1TReEmulFromRAW2016(process):
    process.load('L1Trigger.Configuration.SimL1Emulator_cff')
    process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
    process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('hcalDigis'),
        cms.InputTag('hcalDigis')
    )
    process.simHcalTriggerPrimitiveDigis.inputUpgradeLabel = cms.VInputTag(
                cms.InputTag('hcalDigis'),
                cms.InputTag('hcalDigis')
    )
    process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi')
    process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )  
    process.L1TReEmul = cms.Sequence(process.simEcalTriggerPrimitiveDigis * process.simHcalTriggerPrimitiveDigis * process.SimL1Emulator)
    if stage2L1Trigger.isChosen():
        #cutlist=['simDtTriggerPrimitiveDigis','simCscTriggerPrimitiveDigis']
        #for b in cutlist:
        #    process.SimL1Emulator.remove(getattr(process,b))
        # TwinMux
        process.simTwinMuxDigis.RPC_Source         = cms.InputTag('RPCTwinMuxRawToDigi')
        process.simTwinMuxDigis.DTDigi_Source      = cms.InputTag('twinMuxStage2Digis:PhIn')
        process.simTwinMuxDigis.DTThetaDigi_Source = cms.InputTag('twinMuxStage2Digis:ThIn')
        # BMTF
        process.simBmtfDigis.DTDigi_Source         = cms.InputTag('simTwinMuxDigis')
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
        return process
    else:
        process.simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'ecalDigis:EcalTriggerPrimitives' ) )
        process.simRctDigis.hcalDigis = cms.VInputTag('hcalDigis:')
        process.simRpcTriggerDigis.label         = 'muonRPCDigis'
        process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
        process.schedule.append(process.L1TReEmulPath)
        return process

def L1TReEmulFromRAW(process):
    L1TReEmulFromRAW2016(process)

    if stage2L1Trigger_2017.isChosen():
        process.simOmtfDigis.srcRPC                = cms.InputTag('omtfStage2Digis')
        process.simOmtfDigis.srcCSC                = cms.InputTag('omtfStage2Digis')
        process.simOmtfDigis.srcDTPh               = cms.InputTag('omtfStage2Digis')
        process.simOmtfDigis.srcDTTh               = cms.InputTag('omtfStage2Digis')

    print("# L1TReEmul sequence:  ")
    print("# {0}".format(process.L1TReEmul))
    print("# {0}".format(process.schedule))
    return process

def L1TReEmulFromRAWCalouGT(process):
    L1TReEmulFromRAW(process)
    process.simGtStage2Digis.MuonInputTag   = cms.InputTag("gtStage2Digis","Muon")
    return process 

def L1TReEmulFromNANO(process):

    process.load('L1Trigger.Configuration.SimL1Emulator_cff')
    process.L1TReEmul = cms.Sequence(process.SimL1TGlobal)
    if stage2L1Trigger_2017.isChosen():
        process.simGtStage2Digis.ExtInputTag = cms.InputTag("hltGtStage2Digis")
        process.simGtStage2Digis.MuonInputTag = cms.InputTag("hltGtStage2Digis", "Muon")
        process.simGtStage2Digis.EtSumInputTag = cms.InputTag("hltGtStage2Digis", "EtSum")
        process.simGtStage2Digis.EGammaInputTag = cms.InputTag("hltGtStage2Digis", "EGamma")
        process.simGtStage2Digis.TauInputTag = cms.InputTag("hltGtStage2Digis", "Tau")
        process.simGtStage2Digis.JetInputTag = cms.InputTag("hltGtStage2Digis", "Jet")
        
    process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
    process.schedule.append(process.L1TReEmulPath)

    print ("# L1TReEmul sequence:  ")
    print ("# {0}".format(process.L1TReEmul))
    print ("# {0}".format(process.schedule))
    return process 

def L1TReEmulFromRAWCalo(process):
    process.load('L1Trigger.Configuration.SimL1CaloEmulator_cff')
    process.L1TReEmul = cms.Sequence(process.SimL1CaloEmulator)
    process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag('ecalDigis:EcalTriggerPrimitives')
    process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag('hcalDigis:')
    process.L1TReEmulPath = cms.Path(process.L1TReEmul)
    process.schedule.append(process.L1TReEmulPath)

    print ("# L1TReEmul sequence:  ")
    print ("# {0}".format(process.L1TReEmul))
    print ("# {0}".format(process.schedule))
    return process

def L1TReEmulMCFromRAW(process):
    L1TReEmulFromRAW(process)
    if stage2L1Trigger.isChosen():
            process.simEmtfDigis.CSCInput           = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED')
            process.simOmtfDigis.srcCSC             = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED')
    return process

def L1TReEmulMCFromRAWSimEcalTP(process):
    L1TReEmulMCFromRAW(process)
    if stage2L1Trigger.isChosen():
            process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
    return process

def L1TReEmulMCFromRAWSimHcalTP(process):
    L1TReEmulMCFromRAW(process)
    if stage2L1Trigger.isChosen():
            process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')
    return process

def L1TReEmulMCFrom90xRAWSimHcalTP(process):
    L1TReEmulMCFromRAW(process)
    if stage2L1Trigger.isChosen():
            process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
                cms.InputTag('simHcalUnsuppressedDigis'),
                cms.InputTag('simHcalUnsuppressedDigis')
            )
            process.simHcalTriggerPrimitiveDigis.inputUpgradeLabel = cms.VInputTag(
                cms.InputTag('simHcalUnsuppressedDigis:HBHEQIE11DigiCollection'),
                cms.InputTag('simHcalUnsuppressedDigis:HFQIE10DigiCollection')
            )
            process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')
    return process
    #inputUpgradeLabel = cms.VInputTag(
    #    cms.InputTag('simHcalUnsuppressedDigis:HBHEQIE11DigiCollection'),
    #    cms.InputTag('simHcalUnsuppressedDigis:HFQIE10DigiCollection')),

def L1TReEmulMCFromRAWSimCalTP(process):
    L1TReEmulMCFromRAW(process)
    if stage2L1Trigger.isChosen():
            process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
            process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')
    return process

def L1TReEmulFromRAWsimEcalTP(process):
    L1TReEmulFromRAW(process)
    if stage2L1Trigger.isChosen():
            process.simCaloStage2Layer1Digis.ecalToken = cms.InputTag("simEcalTriggerPrimitiveDigis")
    return process

def L1TReEmulFromRAWsimHcalTP(process):
    L1TReEmulFromRAW(process)
    if stage2L1Trigger.isChosen():
            process.simCaloStage2Layer1Digis.hcalToken = cms.InputTag('simHcalTriggerPrimitiveDigis')
    return process

def L1TReEmulFromRAWsimTP(process):
    L1TReEmulFromRAW(process)
    if stage2L1Trigger.isChosen():
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

def L1TReEmulFromRAWLegacyMuon(process):
    process.load('L1Trigger.Configuration.SimL1Emulator_cff')
    process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
    process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('hcalDigis'),
        cms.InputTag('hcalDigis')
    )

## - Legacy to upgrade format muon converter
    process.load('L1Trigger.L1TCommon.muonLegacyInStage2FormatDigis_cfi')
    process.muonLegacyInStage2FormatDigis.muonSource = cms.InputTag('simGmtDigis')  

## - DT TP emulator
    from L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi import dtTriggerPrimitiveDigis
    process.simDtTriggerPrimitiveDigis = dtTriggerPrimitiveDigis.clone()
    process.simDtTriggerPrimitiveDigis.digiTag = cms.InputTag('muonDTDigis')

## - TwinMux
    from L1Trigger.L1TTwinMux.simTwinMuxDigis_cfi import simTwinMuxDigis
    process.simTwinMuxDigisForDttf = simTwinMuxDigis.clone()
    process.simTwinMuxDigisForDttf.RPC_Source         = cms.InputTag('muonRPCDigis')
    process.simTwinMuxDigisForDttf.DTDigi_Source      = cms.InputTag('bmtfDigis')
    process.simTwinMuxDigisForDttf.DTThetaDigi_Source = cms.InputTag('bmtfDigis')

## - CSC TP emulator 
    from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi import cscTriggerPrimitiveDigis
    process.simCscTriggerPrimitiveDigis = cscTriggerPrimitiveDigis.clone()
    process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi' )
    process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )
#
# - CSC Track Finder emulator
#
    from L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi import csctfTrackDigis
    process.simCsctfTrackDigis = csctfTrackDigis.clone()
    process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag( 'csctfDigis' ) 
    process.simCsctfTrackDigis.DTproducer = 'simDtTriggerPrimitiveDigis'
    from L1Trigger.CSCTrackFinder.csctfDigis_cfi import csctfDigis
    process.simCsctfDigis = csctfDigis.clone()
    process.simCsctfDigis.CSCTrackProducer = 'simCsctfTrackDigis'
##
## - DT Track Finder emulator
## 
    from L1Trigger.DTTrackFinder.dttfDigis_cfi import dttfDigis
    process.simDttfDigis = dttfDigis.clone()
    process.simDttfDigis.DTDigi_Source  = 'simTwinMuxDigisForDttf'
    process.simDttfDigis.CSCStub_Source = 'simCsctfTrackDigis'
##
## - RPC PAC Trigger emulator
##
    from L1Trigger.RPCTrigger.rpcTriggerDigis_cff import rpcTriggerDigis
    process.load('L1Trigger.RPCTrigger.RPCConeConfig_cff')
    process.simRpcTriggerDigis = rpcTriggerDigis.clone()
    process.simRpcTriggerDigis.label = 'muonRPCDigis'
    process.simRpcTriggerDigis.RPCTriggerDebug = cms.untracked.int32(1)

## 
## - Legacy Global Muon Trigger emulator
##
    from L1Trigger.GlobalMuonTrigger.gmtDigis_cfi import gmtDigis
    process.simGmtDigis = gmtDigis.clone()
    process.simGmtDigis.DTCandidates   = cms.InputTag( 'simDttfDigis', 'DT' )
    process.simGmtDigis.CSCCandidates  = cms.InputTag( 'simCsctfDigis', 'CSC' )
    process.simGmtDigis.RPCbCandidates = cms.InputTag( 'simRpcTriggerDigis', 'RPCb' )
    process.simGmtDigis.RPCfCandidates = cms.InputTag( 'simRpcTriggerDigis', 'RPCf' )


    # This is for the upgrade

    # BMTF
    process.simBmtfDigis.DTDigi_Source         = cms.InputTag('bmtfDigis')
    process.simBmtfDigis.DTDigi_Theta_Source   = cms.InputTag('bmtfDigis')
    # TwinMux
    process.simTwinMuxDigis.RPC_Source         = cms.InputTag('muonRPCDigis')
    process.simTwinMuxDigis.DTDigi_Source      = cms.InputTag('bmtfDigis')
    process.simTwinMuxDigis.DTThetaDigi_Source = cms.InputTag('bmtfDigis')
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
   

# - Sequences 
    process.L1TMuonTriggerPrimitives = cms.Sequence(process.simCscTriggerPrimitiveDigis + process.simDtTriggerPrimitiveDigis + process.simTwinMuxDigisForDttf)

    process.L1TReEmul = cms.Sequence(process.L1TMuonTriggerPrimitives + process.simCsctfTrackDigis + process.simCsctfDigis + process.simDttfDigis + process.simRpcTriggerDigis + process.simGmtDigis + process.muonLegacyInStage2FormatDigis)
    
    process.load('L1Trigger.L1TMuon.simMuonQualityAdjusterDigis_cfi')

    process.L1TReEmul = cms.Sequence( process.L1TReEmul + process.simTwinMuxDigis + process.simBmtfDigis + process.simEmtfDigis + process.simOmtfDigis + process.simGmtCaloSumDigis + process.simMuonQualityAdjusterDigis + process.simGmtStage2Digis)

    process.L1TReEmul = cms.Sequence( process.L1TReEmul + process.SimL1TechnicalTriggers + process.SimL1TGlobal )

    process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
    process.schedule.append(process.L1TReEmulPath)
    print("# L1TReEmul sequence:  ")
    print("# {0}".format(process.L1TReEmul))
    print("# {0}".format(process.schedule))
    return process

