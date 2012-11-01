import FWCore.ParameterSet.Config as cms

SiPixelFakeLorentzAngleESSource = cms.ESSource("SiPixelFakeLorentzAngleESSource",
    file = cms.FileInPath('CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt'),
    tanLorentzAnglePerTesla = cms.double(0.106)
)


