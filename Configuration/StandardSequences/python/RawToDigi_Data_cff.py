import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_cff import *


ecalDigis.DoRegional = False
#False by default ecalDigis.DoRegional = False

# RPC Merged Digis 
_muonRPCDigis = muonRPCDigis.copy()
from Configuration.Eras.Modifier_run3_RPC_cff import run3_RPC
_muonRPCDigis.inputTagTwinMuxDigis = cms.InputTag('rpcTwinMuxRawToDigi')
_muonRPCDigis.inputTagOMTFDigis = cms.InputTag('omtfStage2Digis')
_muonRPCDigis.inputTagCPPFDigis = cms.InputTag('rpcCPPFRawToDigi')
run3_RPC.toReplaceWith(muonRPCDigis,_muonRPCDigis)
