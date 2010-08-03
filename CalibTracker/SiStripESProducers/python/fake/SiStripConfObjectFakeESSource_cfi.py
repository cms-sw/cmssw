import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripConfObjectGeneratorService_cfi import *

siStripConfObjectFakeESSource = cms.ESSource("SiStripConfObjectFakeESSource",
                                         appendToDataLabel = cms.string('')
                                         )



