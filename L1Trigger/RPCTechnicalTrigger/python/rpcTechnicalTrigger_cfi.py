import FWCore.ParameterSet.Config as cms

trigmode  = 1

rpcTechnicalTrigger  = cms.EDProducer('RPCTechnicalTrigger',
                                      TriggerMode = cms.int32(trigmode),
                                      RPCDigiLabel = cms.InputTag("muonRPCDigis"),
                                      BitNumbers=cms.vuint32(24,25,26,27,28),
                                      BitNames=cms.vstring('L1Tech_rpcBit1',
                                                           'L1Tech_rpcBit2',
                                                           'L1Tech_rpcBit3',
                                                           'L1Tech_rpcBit4',
                                                           'L1Tech_rpcBit5') )


