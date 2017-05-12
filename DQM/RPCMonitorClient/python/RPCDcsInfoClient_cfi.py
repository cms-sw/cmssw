import FWCore.ParameterSet.Config as cms

rpcDcsInfoClient = cms.EDProducer("RPCDcsInfoClient",
                                  dcsInfoFolder = cms.untracked.string("RPC/DCSInfo")
                                  )
