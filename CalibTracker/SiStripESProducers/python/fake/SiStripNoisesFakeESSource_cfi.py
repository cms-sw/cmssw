import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripNoisesGeneratorService_cfi import *

siStripNoisesFakeESSource = cms.ESSource("SiStripNoisesFakeESSource",
                                         appendToDataLabel = cms.string('')
                                         )



