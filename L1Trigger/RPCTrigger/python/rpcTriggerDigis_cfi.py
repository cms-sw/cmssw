import FWCore.ParameterSet.Config as cms

# Module to generate response of L1Trigger/RPCTrigger
rpcTriggerDigis = cms.EDFilter("RPCTrigger",
    buildOwnLinkSystem = cms.bool(False),
    RPCTriggerDebug = cms.untracked.int32(0),
    label = cms.string('muonRPCDigis')
)


