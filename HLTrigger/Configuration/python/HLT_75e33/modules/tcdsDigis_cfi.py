import FWCore.ParameterSet.Config as cms

tcdsDigis = cms.EDProducer("TcdsRawToDigi",
    InputLabel = cms.InputTag("rawDataCollector"),
    mightGet = cms.optional.untracked.vstring
)
