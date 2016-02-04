import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripLorentzAngleGeneratorService_cfi import *

siStripLorentzAngleFakeESSource = cms.ESSource("SiStripLorentzAngleFakeESSource",
                                               appendToDataLabel = cms.string('')
                                               )



