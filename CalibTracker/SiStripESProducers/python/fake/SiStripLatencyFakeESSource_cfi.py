import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripLatencyGeneratorService_cfi import *

siStripLatencyFakeESSource = cms.ESSource("SiStripLatencyFakeESSource",
                                         appendToDataLabel = cms.string('')
                                         )



