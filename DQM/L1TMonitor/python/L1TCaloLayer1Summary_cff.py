import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM

from L1Trigger.Configuration.SimL1Emulator_cff import *
from L1Trigger.Configuration.CaloTriggerPrimitives_cff import *
simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
    cms.InputTag('hcalDigis'),
    cms.InputTag('hcalDigis')
)
simHcalTriggerPrimitiveDigis.inputUpgradeLabel = cms.VInputTag(
            cms.InputTag('hcalDigis'),
            cms.InputTag('hcalDigis')
)
simDtTriggerPrimitiveDigis.digiTag = cms.InputTag("muonDTDigis")
simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi')
simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )  
L1TReEmul = cms.Sequence(simEcalTriggerPrimitiveDigis * simHcalTriggerPrimitiveDigis * SimL1Emulator)

from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Summary_cfi import simCaloStage2Layer1Summary as _simCaloStage2Layer1Summary
cicadaEmulFromDigis = _simCaloStage2Layer1Summary.clone(caloLayer1Regions = cms.InputTag("caloLayer1Digis", ""))
L1TReEmul.replace(simCaloStage2Layer1Summary, cicadaEmulFromDigis)

# TwinMux
stage2L1Trigger.toModify(simTwinMuxDigis,
    RPC_Source         = 'rpcTwinMuxRawToDigi',
    DTDigi_Source      = 'twinMuxStage2Digis:PhIn',
    DTThetaDigi_Source = 'twinMuxStage2Digis:ThIn'
)
# BMTF
stage2L1Trigger.toModify(simBmtfDigis,
DTDigi_Source         = "simTwinMuxDigis",
DTDigi_Theta_Source   = "bmtfDigis"
)
# KBMTF
stage2L1Trigger.toModify(simKBmtfStubs,
srcPhi       = 'simTwinMuxDigis',
srcTheta     = 'bmtfDigis'
)
# OMTF
stage2L1Trigger.toModify(simOmtfDigis,
    srcRPC  = 'muonRPCDigis',
    srcCSC  = 'csctfDigis',
    srcDTPh = 'bmtfDigis',
    srcDTTh = 'bmtfDigis'
)
# EMTF
stage2L1Trigger.toModify(simEmtfDigis,
    CSCInput = 'emtfStage2Digis',
    RPCInput = 'muonRPCDigis'
)
# Calo Layer1
stage2L1Trigger.toModify(simCaloStage2Layer1Digis,
    ecalToken = 'ecalDigis:EcalTriggerPrimitives',
    hcalToken = 'hcalDigis:'
)

(~stage2L1Trigger).toModify(simRctDigis,
    ecalDigis = ['ecalDigis:EcalTriggerPrimitives'],
    hcalDigis = ['hcalDigis:']
)
(~stage2L1Trigger).toModify(simRpcTriggerDigis, label = 'muonRPCDigis')

# if not hasattr(process, 'L1TReEmulPath'):
#     process.L1TReEmulPath = cms.Path(process.L1TReEmul)    
#     process.schedule.append(process.L1TReEmulPath)

stage2L1Trigger_2017.toModify(simOmtfDigis,
    srcRPC   = 'omtfStage2Digis',
    srcCSC   = 'omtfStage2Digis',
    srcDTPh  = 'omtfStage2Digis',
    srcDTTh  = 'omtfStage2Digis'
)

stage2L1Trigger.toModify(simEmtfDigis,
    CSCInput  = cms.InputTag('emtfStage2Digis'),
    RPCInput  = cms.InputTag('muonRPCDigis'),
    CPPFInput = cms.InputTag('emtfStage2Digis'),
    GEMEnable = cms.bool(False),
    GEMInput  = cms.InputTag('muonGEMPadDigis'),
    CPPFEnable = cms.bool(True), # Use CPPF-emulated clustered RPC hits from CPPF as the RPC hits
)

run3_GEM.toModify(simMuonGEMPadDigis,
    InputCollection         = 'muonGEMDigis',
)

run3_GEM.toModify(simTwinMuxDigis,
    RPC_Source         = 'rpcTwinMuxRawToDigi',
    DTDigi_Source      = 'simDtTriggerPrimitiveDigis',
    DTThetaDigi_Source = 'simDtTriggerPrimitiveDigis'
)

run3_GEM.toModify(simKBmtfStubs,
    srcPhi   = 'bmtfDigis',
    srcTheta = 'bmtfDigis'
)

run3_GEM.toModify(simBmtfDigis,
    DTDigi_Source       = 'bmtfDigis',
    DTDigi_Theta_Source = 'bmtfDigis'
)

from DQM.L1TMonitor.L1TCaloLayer1Summary_cfi import l1tCaloLayer1Summary as _l1tCaloLayer1Summary
l1tCaloLayer1Summary = _l1tCaloLayer1Summary.clone(simCICADAScore = cms.InputTag("cicadaEmulFromDigis", "CICADAScore"))
l1tCaloLayer1SummarySeq = cms.Sequence(L1TReEmul * l1tCaloLayer1Summary)