import FWCore.ParameterSet.Config as cms

siPixelQualityESProducer = cms.ESProducer("SiPixelQualityESProducer",
    siPixelQualityLabel = cms.string(""),
)
