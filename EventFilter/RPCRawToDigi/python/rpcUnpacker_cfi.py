import FWCore.ParameterSet.Config as cms

rpcunpacker = cms.EDFilter("RPCUnpackingModule",
    InputLabel = cms.untracked.InputTag("source")
)


