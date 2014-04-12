import FWCore.ParameterSet.Config as cms

rpcDcsInfoClient = cms.EDAnalyzer("RPCDcsInfoClient",
                                  dcsInfoFolder = cms.untracked.string("RPC/DCSInfo")
                                  )
