import FWCore.ParameterSet.Config as cms


SiStripNoisesGenerator = cms.Service("SiStripNoisesGenerator",
                                     printDebug = cms.untracked.uint32(5),
                                     file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
                                                                          
                                     StripLengthMode = cms.bool(True),

                                     #relevant if striplenght mode is chosen
                                     # standard value for deconvolution mode is 51. For peak mode 38.8.
                                     # standard value for deconvolution mode is 630. For peak mode  414.

                                     # TIB
                                     NoiseStripLengthSlopeTIB = cms.vdouble( 51.0,  51.0,  51.0,  51.0),
                                     NoiseStripLengthQuoteTIB = cms.vdouble(630.0, 630.0, 630.0, 630.0),
                                     # TID                         
                                     NoiseStripLengthSlopeTID = cms.vdouble( 51.0,  51.0,  51.0),
                                     NoiseStripLengthQuoteTID = cms.vdouble(630.0, 630.0, 630.0),
                                     # TOB                         
                                     NoiseStripLengthSlopeTOB = cms.vdouble( 51.0,  51.0,  51.0,  51.0,  51.0,  51.0),
                                     NoiseStripLengthQuoteTOB = cms.vdouble(630.0, 630.0, 630.0, 630.0, 630.0, 630.0),
                                     # TEC
                                     NoiseStripLengthSlopeTEC = cms.vdouble( 51.0,  51.0,  51.0,  51.0,  51.0,  51.0,  51.0),
                                     NoiseStripLengthQuoteTEC = cms.vdouble(630.0, 630.0, 630.0, 630.0, 630.0, 630.0, 630.0),

                                     electronPerAdc = cms.double(1.0),

                                     #relevant if random mode is chosen
                                     # TIB
                                     MeanNoiseTIB  = cms.vdouble(4.0, 4.0, 4.0, 4.0),
                                     SigmaNoiseTIB = cms.vdouble(0.5, 0.5, 0.5, 0.5),
                                     # TID
                                     MeanNoiseTID  = cms.vdouble(4.0, 4.0, 4.0),
                                     SigmaNoiseTID = cms.vdouble(0.5, 0.5, 0.5),
                                     # TOB
                                     MeanNoiseTOB  = cms.vdouble(4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
                                     SigmaNoiseTOB = cms.vdouble(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                     # TEC
                                     MeanNoiseTEC  = cms.vdouble(4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
                                     SigmaNoiseTEC = cms.vdouble(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),

                                     MinPositiveNoise = cms.double(0.1)
                                     )

from SimGeneral.MixingModule.stripDigitizer_cfi import *
#from SimTracker.SiStripDigitizer.SiStripDigi_cfi import *
SiStripNoisesGenerator.electronPerAdc=stripDigitizer.electronPerAdcDec


