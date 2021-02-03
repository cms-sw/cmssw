import FWCore.ParameterSet.Config as cms

scalersRawToDigi = cms.EDProducer("ScalersRawToDigi",
    mightGet = cms.optional.untracked.vstring,
    scalersInputTag = cms.InputTag("rawDataCollector")
)
