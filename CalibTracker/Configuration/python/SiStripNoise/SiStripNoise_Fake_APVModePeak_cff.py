import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.fake.SiStripNoisesFakeESSource_cfi import *    
SiStripNoisesGenerator.NoiseStripLengthSlope = cms.vdouble(38.8)
SiStripNoisesGenerator.NoiseStripLengthQuote = cms.vdouble(414.)

