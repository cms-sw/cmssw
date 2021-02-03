import FWCore.ParameterSet.Config as cms

muonRPCDigis = cms.EDProducer("RPCUnpackingModule",
    InputLabel = cms.InputTag("rawDataCollector"),
    doSynchro = cms.bool(True),
    mightGet = cms.optional.untracked.vstring
)
