import FWCore.ParameterSet.Config as cms

# Stage2 fake Technical Triggers
import L1Trigger.L1TGlobal.simGtExtFakeProd_cfi
simGtExtFakeStage2Digis = L1Trigger.L1TGlobal.simGtExtFakeProd_cfi.simGtExtFakeProd.clone()

SimL1TechnicalTriggers = cms.Sequence(simGtExtFakeStage2Digis)


# BSC Technical Trigger
import L1TriggerOffline.L1Analyzer.bscTrigger_cfi
simBscDigis = L1TriggerOffline.L1Analyzer.bscTrigger_cfi.bscTrigger.clone()

# RPC Technical Trigger
import L1Trigger.RPCTechnicalTrigger.rpcTechnicalTrigger_cfi
simRpcTechTrigDigis = L1Trigger.RPCTechnicalTrigger.rpcTechnicalTrigger_cfi.rpcTechnicalTrigger.clone()

simRpcTechTrigDigis.RPCDigiLabel = 'simMuonRPCDigis'

# HCAL Technical Trigger
import SimCalorimetry.HcalTrigPrimProducers.hcalTTPRecord_cfi
simHcalTechTrigDigis = SimCalorimetry.HcalTrigPrimProducers.hcalTTPRecord_cfi.simHcalTTPRecord.clone()

# CASTOR Techical Trigger
import SimCalorimetry.CastorTechTrigProducer.castorTTRecord_cfi
simCastorTechTrigDigis = SimCalorimetry.CastorTechTrigProducer.castorTTRecord_cfi.simCastorTTRecord.clone()

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
if not (stage2L1Trigger.isChosen() or phase2_common.isChosen()):
    SimL1TechnicalTriggers = cms.Sequence( 
        simBscDigis + 
        simRpcTechTrigDigis + 
        simHcalTechTrigDigis +
        simCastorTechTrigDigis )
