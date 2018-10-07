import FWCore.ParameterSet.Config as cms

rawDataCollector = cms.EDProducer("RawDataMapperByLabel",
    RawCollectionList = cms.VInputTag( cms.InputTag('rawDataReducedFormat'),
                                       cms.InputTag('rawDataRepacker'),
                                       cms.InputTag('rawDataCollector')),
    MainCollection= cms.InputTag('rawDataCollector')                          
)

