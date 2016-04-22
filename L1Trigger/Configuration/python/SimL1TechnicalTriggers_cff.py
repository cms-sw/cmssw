import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

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

if not (eras.stage2L1Trigger.isChosen()):
    SimL1TechnicalTriggers = cms.Sequence( 
        simBscDigis + 
        simRpcTechTrigDigis + 
        simHcalTechTrigDigis +
        simCastorTechTrigDigis )
