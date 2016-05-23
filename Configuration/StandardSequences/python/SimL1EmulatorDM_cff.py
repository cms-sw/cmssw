import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimL1Emulator_cff import *
from Configuration.StandardSequences.Eras import eras

# Modifications for DataMixer input:
simDtTriggerPrimitiveDigis.digiTag = 'mixData'
simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag("mixData","MuonCSCComparatorDigisDM")
simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag("mixData","MuonCSCWireDigisDM")
#
#
simRpcTechTrigDigis.RPCDigiLabel = 'mixData'
#
simHcalTechTrigDigis.ttpDigiCollection = "DMHcalTTPDigis"
#

if not eras.stage2L1Trigger.isChosen():
    simRpcTriggerDigis.label = 'mixData'
    simRctDigis.hcalDigis=cms.VInputTag(cms.InputTag("DMHcalTriggerPrimitiveDigis"))   
    simRctDigis.ecalDigis=cms.VInputTag(cms.InputTag("DMEcalTriggerPrimitiveDigis"))   
else:
    simTwinMuxDigis.RPC_Source = cms.InputTag('mixData')
    simOmtfDigis.srcRPC = cms.InputTag('mixData')
    simCaloStage2Layer1Digis.ecalToken = cms.InputTag("DMEcalTriggerPrimitiveDigis")
    simCaloStage2Layer1Digis.hcalToken = cms.InputTag("DMHcalTriggerPrimitiveDigis")
