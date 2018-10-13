import FWCore.ParameterSet.Config as cms

rawDataMapper = cms.EDProducer("RawDataMapperByLabel",
    rawCollectionList = cms.VInputTag( cms.InputTag('rawDataCollector'),
                                       cms.InputTag('rawDataRepacker'),
                                       cms.InputTag('rawDataReducedFormat')),
    mainCollection= cms.InputTag('rawDataCollector')                          
)

