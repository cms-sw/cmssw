import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripApvGainGeneratorService_cfi import *

siStripApvGainFakeESSource = cms.ESSource("SiStripApvGainFakeESSource",
                                          appendToDataLabel = cms.string('')
                                          )



