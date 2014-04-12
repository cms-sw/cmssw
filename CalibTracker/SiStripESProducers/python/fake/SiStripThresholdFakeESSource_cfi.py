import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripThresholdGeneratorService_cfi import *

siStripThresholdFakeESSource = cms.ESSource("SiStripThresholdFakeESSource",
                                            appendToDataLabel = cms.string('')
                                            )



