import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripBadModuleGeneratorService_cfi import *

siStripBadModuleConfigurableFakeESSource = cms.ESSource("SiStripBadModuleConfigurableFakeESSource",
                                                        appendToDataLabel = cms.string('')
                                                        )



