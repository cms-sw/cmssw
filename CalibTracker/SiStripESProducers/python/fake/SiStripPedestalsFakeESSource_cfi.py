import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripPedestalsGeneratorService_cfi import *

siStripPedestalsFakeESSource = cms.ESSource("SiStripPedestalsFakeESSource",
                                            appendToDataLabel = cms.string('')
                                            )



