from __future__ import print_function

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM

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

    stage2L1Trigger.toModify(process.simTwinMuxDigis,
        RPC_Source         = 'muonRPCDigis',
        # When available, this will switch to TwinMux input Digis:
        DTDigi_Source      = "dttfDigis",
        DTThetaDigi_Source = "dttfDigis"
    )
    stage2L1Trigger.toModify(process.simOmtfDigis,
        srcRPC = 'muonRPCDigis',
        srcCSC = "csctfDigis"
    )
    stage2L1Trigger.toModify(process.simBmtfDigis,
       DTDigi_Source         = "simTwinMuxDigis",
       DTDigi_Theta_Source   = "dttfDigis"
    )
    stage2L1Trigger.toModify(process.simKBmtfStubs,
        srcPhi     = "simTwinMuxDigis",
        srcTheta   = "dttfDigis"
    )
    stage2L1Trigger.toModify(process.simEmtfDigis,
        CSCInput = "csctfDigis",
        RPCInput = 'muonRPCDigis'
    )
    stage2L1Trigger.toModify(process.simCaloStage2Layer1Digis, ecalToken = "ecalDigis:EcalTriggerPrimitives")
    # quiet warning abouts missing Stage-2 payloads, since they won't reliably exist in 2015 data.
    stage2L1Trigger.toModify(process.caloStage2Digis, MinFeds = 0)
    stage2L1Trigger.toModify(process.gmtStage2Digis, MinFeds = 0)
    stage2L1Trigger.toModify(process.gtStage2Digis, MinFeds = 0)

    (~stage2L1Trigger).toModify(process.simRctDigis,
        ecalDigis = ['simEcalTriggerPrimitiveDigis'],
        hcalDigis = ['simHcalTriggerPrimitiveDigis']
    )
    (~stage2L1Trigger).toModify(process.simRpcTriggerDigis, label = 'muonRPCDigis')
    (~stage2L1Trigger).toModify(process.simRpcTechTrigDigis, RPCDigiLabel  = 'muonRPCDigis')

    process.L1TReEmulPath = cms.Path(process.L1TReEmul)
    process.schedule.append(process.L1TReEmulPath)

    print("# L1TReEmul sequence:  ")
    print("# {0}".format(process.L1TReEmul))
    print("# {0}".format(process.schedule))
    return process

def L1TReEmulMCFromRAW2015(process):
    L1TReEmulFromRAW2015(process)
    stage2L1Trigger.toModify(process.simEmtfDigis, CSCInput = 'simCscTriggerPrimitiveDigis:MPCSORTED')
    stage2L1Trigger.toModify(process.simOmtfDigis, srcCSC   = 'simCscTriggerPrimitiveDigis:MPCSORTED')
    return process

def L1TReEmulFromRAW2015simCaloTP(process):
    L1TReEmulFromRAW2015(process)
    stage2L1Trigger.toModify(process.simCaloStage2Layer1Digis, ecalToken = "simEcalTriggerPrimitiveDigis")
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
    process.simDtTriggerPrimitiveDigis.digiTag = cms.InputTag("muonDTDigis")
    process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi')
    process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )  
    process.L1TReEmul = cms.Sequence(process.simEcalTriggerPrimitiveDigis * process.simHcalTriggerPrimitiveDigis * process.SimL1Emulator)


    #cutlist=['simDtTriggerPrimitiveDigis','simCscTriggerPrimitiveDigis']
    #for b in cutlist:
    #    process.SimL1Emulator.remove(getattr(process,b))
    # TwinMux
    stage2L1Trigger.toModify(process.simTwinMuxDigis,
        RPC_Source         = 'rpcTwinMuxRawToDigi',
        DTDigi_Source      = 'twinMuxStage2Digis:PhIn',
        DTThetaDigi_Source = 'twinMuxStage2Digis:ThIn'
    )
    # BMTF
    stage2L1Trigger.toModify(process.simBmtfDigis,
       DTDigi_Source         = "simTwinMuxDigis",
       DTDigi_Theta_Source   = "bmtfDigis"
    )
    # KBMTF
    stage2L1Trigger.toModify(process.simKBmtfStubs,
       srcPhi       = 'simTwinMuxDigis',
       srcTheta     = 'bmtfDigis'
    )
    # OMTF
    stage2L1Trigger.toModify(process.simOmtfDigis,
        srcRPC  = 'muonRPCDigis',
        srcCSC  = 'csctfDigis',
        srcDTPh = 'bmtfDigis',
        srcDTTh = 'bmtfDigis'
    )
    # EMTF
    stage2L1Trigger.toModify(process.simEmtfDigis,
        CSCInput = 'emtfStage2Digis',
        RPCInput = 'muonRPCDigis'
    )
    # Calo Layer1
    stage2L1Trigger.toModify(process.simCaloStage2Layer1Digis,
        ecalToken = 'ecalDigis:EcalTriggerPrimitives',
        hcalToken = 'hcalDigis:'
    )

    (~stage2L1Trigger).toModify(process.simRctDigis,
        ecalDigis = ['ecalDigis:EcalTriggerPrimitives'],
        hcalDigis = ['hcalDigis:']
    )
    (~stage2L1Trigger).toModify(process.simRpcTriggerDigis, label = 'muonRPCDigis')

    process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
    process.schedule.append(process.L1TReEmulPath)
    return process 

def L1TReEmulFromRAW(process):
    L1TReEmulFromRAW2016(process)
    

    stage2L1Trigger_2017.toModify(process.simOmtfDigis,
        srcRPC   = 'omtfStage2Digis',
        srcCSC   = 'omtfStage2Digis',
        srcDTPh  = 'omtfStage2Digis',
        srcDTTh  = 'omtfStage2Digis'
    )

    stage2L1Trigger.toModify(process.simEmtfDigis,
      CSCInput  = cms.InputTag('emtfStage2Digis'),
      RPCInput  = cms.InputTag('muonRPCDigis'),
      CPPFInput = cms.InputTag('emtfStage2Digis'),
      GEMEnable = cms.bool(False),
      GEMInput  = cms.InputTag('muonGEMPadDigis'),
      CPPFEnable = cms.bool(True), # Use CPPF-emulated clustered RPC hits from CPPF as the RPC hits
    )

    run3_GEM.toModify(process.simMuonGEMPadDigis,
        InputCollection         = 'muonGEMDigis',
    )

    run3_GEM.toModify(process.simTwinMuxDigis,
        RPC_Source         = 'rpcTwinMuxRawToDigi',
        DTDigi_Source      = 'simDtTriggerPrimitiveDigis',
        DTThetaDigi_Source = 'simDtTriggerPrimitiveDigis'
    )

    run3_GEM.toModify(process.simKBmtfStubs,
        srcPhi   = 'bmtfDigis',
        srcTheta = 'bmtfDigis'
    )

    run3_GEM.toModify(process.simBmtfDigis,
        DTDigi_Source       = 'bmtfDigis',
        DTDigi_Theta_Source = 'bmtfDigis'
    )

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
    stage2L1Trigger_2017.toModify(process.simGtStage2Digis,
        ExtInputTag = "hltGtStage2Digis",
        MuonInputTag = "hltGtStage2Digis:Muon",
        EtSumInputTag = "hltGtStage2Digis:EtSum",
        EGammaInputTag = "hltGtStage2Digis:EGamma",
        TauInputTag = "hltGtStage2Digis:Tau",
        JetInputTag = "hltGtStage2Digis:Jet"
    )
        
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

def L1TReEmulFromRAWCaloSimTP(process):
    process.load('L1Trigger.Configuration.SimL1CaloEmulator_cff')
    process.L1TReEmul = cms.Sequence(process.SimL1CaloEmulator)
    process.L1TReEmulPath = cms.Path(process.L1TReEmul)
    process.schedule.append(process.L1TReEmulPath)

    print ("# L1TReEmul sequence:  ")
    print ("# {0}".format(process.L1TReEmul))
    print ("# {0}".format(process.schedule))
    return process

def L1TReEmulMCFromRAW(process):
    L1TReEmulFromRAW(process)
    stage2L1Trigger.toModify(process.simEmtfDigis, CSCInput = 'simCscTriggerPrimitiveDigis:MPCSORTED')
    stage2L1Trigger.toModify(process.simOmtfDigis, srcCSC   = 'simCscTriggerPrimitiveDigis:MPCSORTED')

    # Temporary fix for OMTF inputs in MC re-emulation
    run3_GEM.toModify(process.simOmtfDigis,
        srcRPC   = 'muonRPCDigis',
        srcDTPh  = 'simDtTriggerPrimitiveDigis',
        srcDTTh  = 'simDtTriggerPrimitiveDigis'
    )

    return process

def L1TReEmulMCFromRAWSimEcalTP(process):
    L1TReEmulMCFromRAW(process)
    stage2L1Trigger.toModify(process.simCaloStage2Layer1Digis, ecalToken = "simEcalTriggerPrimitiveDigis")
    return process

def L1TReEmulMCFromRAWSimHcalTP(process):
    L1TReEmulMCFromRAW(process)
    stage2L1Trigger.toModify(process.simCaloStage2Layer1Digis, hcalToken = 'simHcalTriggerPrimitiveDigis')
    return process

def L1TReEmulMCFrom90xRAWSimHcalTP(process):
    L1TReEmulMCFromRAW(process)
    stage2L1Trigger.toModify(process.simHcalTriggerPrimitiveDigis,
        inputLabel = [
            'simHcalUnsuppressedDigis',
            'simHcalUnsuppressedDigis'
        ],
        inputUpgradeLabel = [
            'simHcalUnsuppressedDigis:HBHEQIE11DigiCollection',
            'simHcalUnsuppressedDigis:HFQIE10DigiCollection'
        ]
    )
    stage2L1Trigger.toModify(process.simCaloStage2Layer1Digis, hcalToken = 'simHcalTriggerPrimitiveDigis')
    return process
    #inputUpgradeLabel = cms.VInputTag(
    #    cms.InputTag('simHcalUnsuppressedDigis:HBHEQIE11DigiCollection'),
    #    cms.InputTag('simHcalUnsuppressedDigis:HFQIE10DigiCollection')),

def L1TReEmulMCFromRAWSimCalTP(process):
    L1TReEmulMCFromRAW(process)
    stage2L1Trigger.toModify(process.simCaloStage2Layer1Digis,
        ecalToken = "simEcalTriggerPrimitiveDigis",
        hcalToken = 'simHcalTriggerPrimitiveDigis'
    ) 
    return process

def L1TReEmulFromRAWsimEcalTP(process):
    L1TReEmulFromRAW(process)
    stage2L1Trigger.toModify(process.simCaloStage2Layer1Digis, ecalToken = "simEcalTriggerPrimitiveDigis")
    return process

def L1TReEmulFromRAWsimHcalTP(process):
    L1TReEmulFromRAW(process)
    stage2L1Trigger.toModify(process.simCaloStage2Layer1Digis, hcalToken = 'simHcalTriggerPrimitiveDigis')
    return process

def L1TReEmulFromRAWsimTP(process):
    L1TReEmulFromRAW(process)
    # TwinMux
    stage2L1Trigger.toModify(process.simTwinMuxDigis,
        RPC_Source         = 'muonRPCDigis',
        DTDigi_Source      = 'simDtTriggerPrimitiveDigis',
        DTThetaDigi_Source = 'simDtTriggerPrimitiveDigis'
    )
    # BMTF
    stage2L1Trigger.toModify(process.simBmtfDigis,
        DTDigi_Source         = 'simTwinMuxDigis',
        DTDigi_Theta_Source   = 'simDtTriggerPrimitiveDigis'
    )
    # KBMTF
    stage2L1Trigger.toModify(process.simKBmtfStubs,
        srcPhi     = "simTwinMuxDigis",
        srcTheta   = "simDtTriggerPrimitiveDigis"
    )
    # OMTF
    stage2L1Trigger.toModify(process.simOmtfDigis,
        srcRPC  = 'muonRPCDigis',
        srcCSC  = 'simCscTriggerPrimitiveDigis',
        srcDTPh = 'simDtTriggerPrimitiveDigis',
        srcDTTh = 'simDtTriggerPrimitiveDigis'
    )
    # EMTF
    stage2L1Trigger.toModify(process.simEmtfDigis,
        CSCInput = 'simCscTriggerPrimitiveDigis',
        RPCInput = 'muonRPCDigis'
    )
    # Layer1
    stage2L1Trigger.toModify(process.simCaloStage2Layer1Digis,
        ecalToken = "simEcalTriggerPrimitiveDigis",
        hcalToken = 'simHcalTriggerPrimitiveDigis'
    )
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

