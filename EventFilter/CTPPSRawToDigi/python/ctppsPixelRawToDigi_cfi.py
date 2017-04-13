import FWCore.ParameterSet.Config as cms


ctppsPixelDigis = cms.EDProducer(
    "CTPPSPixelRawToDigi",
    #    InputLabel = cms.InputTag("ctppsPixelRawData"),
    InputLabel = cms.InputTag("rawDataCollector"),
    mappingLabel = cms.string("RPix")
    
    )
