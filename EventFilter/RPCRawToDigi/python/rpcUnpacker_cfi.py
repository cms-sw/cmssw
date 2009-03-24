import FWCore.ParameterSet.Config as cms

rpcunpacker = cms.EDFilter("RPCUnpackingModule",
    InputLabel = cms.InputTag("source"),
    doSynchro = cms.bool(True)
)


