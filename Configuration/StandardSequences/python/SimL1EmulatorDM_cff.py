import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimL1Emulator_cff import *

# Modifications for DataMixer input:
simDtTriggerPrimitiveDigis.digiTag = 'mixData'
simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag("mixData","MuonCSCComparatorDigisDM")
simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag("mixData","MuonCSCWireDigisDM")
#
#simRpcTriggerDigis.label = cms.InputTag("simMuonRPCDigis")
simRpcTriggerDigis.label = 'mixData'
simRpcTechTrigDigis.RPCDigiLabel = 'mixData'
simHcalTechTrigDigis.ttpDigiCollection = "DMHcalTTPDigis"
simRctDigis.hcalDigis=cms.VInputTag(cms.InputTag("DMHcalTriggerPrimitiveDigis"))
simRctDigis.ecalDigis=cms.VInputTag(cms.InputTag("DMEcalTriggerPrimitiveDigis"))
