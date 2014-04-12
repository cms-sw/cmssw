import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.fake.SiStripNoisesFakeESSource_cfi import *    
SiStripNoisesGenerator.NoiseStripLengthSlope = cms.vdouble(51.)
SiStripNoisesGenerator.NoiseStripLengthQuote = cms.vdouble(630.)

