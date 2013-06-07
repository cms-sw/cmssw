import FWCore.ParameterSet.Config as cms

rpcunpacker = cms.EDProducer("RPCUnpackingModule",
    InputLabel = cms.InputTag("rawDataCollector"),
    doSynchro = cms.bool(True)
)


