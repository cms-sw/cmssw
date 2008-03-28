import FWCore.ParameterSet.Config as cms

SiPixelFakeGainForHLTESSource = cms.ESSource("SiPixelFakeGainForHLTESSource",
    file = cms.FileInPath('CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt')
)


