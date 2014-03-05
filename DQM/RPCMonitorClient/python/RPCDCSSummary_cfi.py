import FWCore.ParameterSet.Config as cms

rpcDCSSummary = cms.EDAnalyzer("RPCDCSSummary", 
                               NumberOfEndcapDisks  = cms.untracked.int32(4),
                               )
