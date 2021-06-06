import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.RawToDigi_cff import *

# RPC Merged Digis 
from Configuration.Eras.Modifier_run3_RPC_cff import run3_RPC
run3_RPC.toModify(muonRPCDigis,
    inputTagTwinMuxDigis = 'rpcTwinMuxRawToDigi',
    inputTagOMTFDigis = 'omtfStage2Digis',
    inputTagCPPFDigis = 'rpcCPPFRawToDigi'
)
