import FWCore.ParameterSet.Config as cms


SiStripNoisesGenerator = cms.Service("SiStripNoisesGenerator",
                                     printDebug = cms.untracked.uint32(5),
                                     file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
                                                                          
                                     StripLengthMode = cms.bool(True),

                                     #relevant if striplenght mode is chosen
                                     # standard value for deconvolution mode is 51. For peak mode 38.8.
                                     # standard value for deconvolution mode is 630. For peak mode  414.
                                     
                                     NoiseStripLengthSlope = cms.double(51.0),
                                     NoiseStripLengthQuote = cms.double(630.0),
                                     electronPerAdc = cms.double(1.0),

                                     #relevant if random mode is chosen
                                     MeanNoise = cms.double(4.0),
                                     SigmaNoise = cms.double(0.5),
                                     MinPositiveNoise = cms.double(0.1)
                                     )

from SimTracker.SiStripDigitizer.SiStripDigi_cfi import *
SiStripNoisesGenerator.electronPerAdc=simSiStripDigis.electronPerAdc


