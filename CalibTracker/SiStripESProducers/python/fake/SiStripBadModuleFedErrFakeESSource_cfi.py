import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripBadModuleFedErrService_cfi import *

siStripBadModuleFedErrFakeESSource = cms.ESSource("SiStripBadModuleFedErrFakeESSource",
                                                   appendToDataLabel = cms.string('')
                                                 )
