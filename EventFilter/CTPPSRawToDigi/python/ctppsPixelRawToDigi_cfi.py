import FWCore.ParameterSet.Config as cms


ctppsPixelDigis = cms.EDProducer(
    "CTPPSPixelRawToDigi",
    InputLabel = cms.InputTag("ctppsPixelRawData"),
    mappingLabel = cms.string("RPix")

)
