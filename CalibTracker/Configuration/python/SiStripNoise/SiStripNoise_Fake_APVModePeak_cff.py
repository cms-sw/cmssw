import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.fake.SiStripNoisesFakeESSource_cfi import siStripNoisesFakeESSource

siStripNoisesFakeESSource.NoiseStripLengthSlope = cms.vdouble(38.8)
siStripNoisesFakeESSource.NoiseStripLengthQuote = cms.vdouble(414.)
