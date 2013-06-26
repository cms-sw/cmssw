import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripBackPlaneCorrectionGeneratorService_cfi import *

siStripBackPlaneCorrectionFakeESSource = cms.ESSource("SiStripBackPlaneCorrectionFakeESSource",
                                               appendToDataLabel = cms.string('')
                                               )
