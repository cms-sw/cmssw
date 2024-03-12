import FWCore.ParameterSet.Config as cms

# Module to generate response of L1Trigger/RPCTrigger
rpcTriggerDigis = cms.EDProducer("RPCTrigger",
    RPCTriggerDebug = cms.untracked.int32(0),
    label = cms.string('muonRPCDigis')
)


# foo bar baz
# mLXdKgQLJAXg5
# OB85HlR1D5uGS
