# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

siStripLatencyFakeESSource = cms.ESSource("SiStripLatencyFakeESSource",
                                         appendToDataLabel = cms.string(''),
                                         latency = cms.uint32(1),
                                         mode = cms.uint32(37)
                                         )
