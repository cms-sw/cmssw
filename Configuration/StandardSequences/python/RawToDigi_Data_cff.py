import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_cff import *


ecalDigis.DoRegional = False
#False by default ecalDigis.DoRegional = False

# RPC Merged Digis 
muonRPCNewDigis.inputTagTwinMuxDigis = cms.InputTag('rpcTwinMuxRawToDigi')
muonRPCNewDigis.inputTagOMTFDigis = cms.InputTag('omtfStage2Digis')
muonRPCNewDigis.inputTagCPPFDigis = cms.InputTag('rpcCPPFRawToDigi')
