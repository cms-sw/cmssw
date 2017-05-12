import FWCore.ParameterSet.Config as cms

rpcDaqInfo = cms.EDProducer("RPCDaqInfo",
                            NumberOfEndcapDisks  = cms.untracked.int32(4)
                            )

