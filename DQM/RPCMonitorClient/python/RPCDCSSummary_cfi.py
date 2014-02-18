import FWCore.ParameterSet.Config as cms

rpcDCSSummary = cms.EDAnalyzer("RPCDCSSummary", 
                               NumberOfEndcapDisks  = cms.untracked.int32(4),
                               MinimumRPCFEDId  = cms.untracked.int32(790),
                               MaximumRPCFEDId  = cms.untracked.int32(792)
                               )
