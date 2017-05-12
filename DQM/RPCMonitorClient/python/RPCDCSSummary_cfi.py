import FWCore.ParameterSet.Config as cms

rpcDCSSummary = cms.EDProducer("RPCDCSSummary", 
                               NumberOfEndcapDisks  = cms.untracked.int32(4),
                               )
