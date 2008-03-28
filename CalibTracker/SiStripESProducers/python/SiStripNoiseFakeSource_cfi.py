import FWCore.ParameterSet.Config as cms

SiStripNoiseFakeESSource = cms.ESSource("SiStripNoiseFakeESSource",
    printDebug = cms.untracked.bool(False),
    # standard value for deconvolution mode is 51. For peak mode 38.8.
    NoiseStripLengthSlope = cms.double(51.0),
    electronPerAdc = cms.double(250.0),
    file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
    # standard value for deconvolution mode is 630. For peak mode  414.
    NoiseStripLengthQuote = cms.double(630.0)
)


