import FWCore.ParameterSet.Config as cms

SiPixelFakeQualityESSource = cms.ESSource("SiPixelFakeQualityESSource",
    file = cms.FileInPath('CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt')
)


