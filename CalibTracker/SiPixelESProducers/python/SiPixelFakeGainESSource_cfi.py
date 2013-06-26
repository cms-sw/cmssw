import FWCore.ParameterSet.Config as cms

SiPixelFakeGainESSource = cms.ESSource("SiPixelFakeGainESSource",
    file = cms.FileInPath('CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt')
)


