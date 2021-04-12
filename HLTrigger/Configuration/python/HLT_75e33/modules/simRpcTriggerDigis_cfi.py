import FWCore.ParameterSet.Config as cms

simRpcTriggerDigis = cms.EDProducer("RPCTrigger",
    RPCTriggerDebug = cms.untracked.int32(0),
    label = cms.string('simMuonRPCDigis')
)
