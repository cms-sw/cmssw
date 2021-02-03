import FWCore.ParameterSet.Config as cms

onlineMetaDataDigis = cms.EDProducer("OnlineMetaDataRawToDigi",
    mightGet = cms.optional.untracked.vstring,
    onlineMetaDataInputLabel = cms.InputTag("rawDataCollector")
)
