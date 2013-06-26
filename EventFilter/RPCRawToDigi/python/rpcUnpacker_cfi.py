import FWCore.ParameterSet.Config as cms

rpcunpacker = cms.EDProducer("RPCUnpackingModule",
    InputLabel = cms.InputTag("source"),
    doSynchro = cms.bool(True)
)


