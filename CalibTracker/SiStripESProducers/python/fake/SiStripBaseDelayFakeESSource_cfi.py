# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

siStripBaseDelayFakeESSource = cms.ESSource("SiStripBaseDelayFakeESSource",
					    appendToDataLabel = cms.string(''),
                                            CoarseDelay = cms.uint32(0),
                                            FineDelay = cms.uint32(0)
					   )
