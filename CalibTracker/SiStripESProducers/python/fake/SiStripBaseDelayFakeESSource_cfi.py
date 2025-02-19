# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.services.SiStripBaseDelayGeneratorService_cfi import *

siStripBaseDelayFakeESSource = cms.ESSource("SiStripBaseDelayFakeESSource",
					    appendToDataLabel = cms.string('')
					   )
