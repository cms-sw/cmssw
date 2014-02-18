import FWCore.ParameterSet.Config as cms

rpcDaqInfo = cms.EDAnalyzer("RPCDaqInfo",
                            NumberOfEndcapDisks  = cms.untracked.int32(4),
                            MinimumRPCFEDId  = cms.untracked.int32(790),
                            MaximumRPCFEDId  = cms.untracked.int32(792)
                            )

