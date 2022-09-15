import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimL1Emulator_cff import *

# In premixing stage2, need to use the original ones for muons
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2

# Modifications for DataMixer input:
(~premix_stage2).toModify(simDtTriggerPrimitiveDigis, digiTag = 'mixData')
(~premix_stage2).toModify(simCscTriggerPrimitiveDigis, CSCComparatorDigiProducer = "mixData:MuonCSCComparatorDigisDM")
(~premix_stage2).toModify(simCscTriggerPrimitiveDigis, CSCWireDigiProducer = "mixData:MuonCSCWireDigisDM")
#
#
(~premix_stage2).toModify(simRpcTechTrigDigis, RPCDigiLabel = 'mixData')
#
simHcalTechTrigDigis.ttpDigiCollection = "DMHcalTTPDigis"
#
l1tHGCalVFEProducer.eeDigis = "mixData:HGCDigisEE"
l1tHGCalVFEProducer.fhDigis = "mixData:HGCDigisHEfront"
l1tHGCalVFEProducer.bhDigis = "mixData:HGCDigisHEback"

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
# Legacy and Stage-1 Trigger
(~premix_stage2).toModify(simRpcTriggerDigis, label = 'mixData')
simRctDigis.hcalDigis=["DMHcalTriggerPrimitiveDigis"]
simRctDigis.ecalDigis=["DMEcalTriggerPrimitiveDigis"]
# Stage-2 Trigger
#seems likely that this code does not support 2015 MC...
(~premix_stage2).toModify(simTwinMuxDigis, RPC_Source = 'mixData')
(~premix_stage2).toModify(simOmtfDigis, srcRPC = 'mixData')
simCaloStage2Layer1Digis.ecalToken = "DMEcalTriggerPrimitiveDigis"
simCaloStage2Layer1Digis.hcalToken = "DMHcalTriggerPrimitiveDigis"
