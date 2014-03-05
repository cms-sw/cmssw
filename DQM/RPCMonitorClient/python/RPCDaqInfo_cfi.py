import FWCore.ParameterSet.Config as cms

rpcDaqInfo = cms.EDAnalyzer("RPCDaqInfo",
                            NumberOfEndcapDisks  = cms.untracked.int32(4)
                            )

