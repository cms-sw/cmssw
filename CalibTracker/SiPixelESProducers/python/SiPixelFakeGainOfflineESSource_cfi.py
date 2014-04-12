import FWCore.ParameterSet.Config as cms

SiPixelFakeGainOfflineESSource = cms.ESSource("SiPixelFakeGainOfflineESSource",
    file = cms.FileInPath('CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt')
)


