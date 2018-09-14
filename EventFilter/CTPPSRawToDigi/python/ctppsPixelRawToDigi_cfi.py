import FWCore.ParameterSet.Config as cms

ctppsPixelDigis = cms.EDProducer("CTPPSPixelRawToDigi",
    #inputLabel = cms.InputTag("ctppsPixelRawData"),
    inputLabel = cms.InputTag("rawDataCollector"),
    mappingLabel = cms.string("RPix") 
    )
