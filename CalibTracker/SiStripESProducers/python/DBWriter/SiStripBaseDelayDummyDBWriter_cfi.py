# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms


siStripBaseDelayDummyDBWriter = cms.EDAnalyzer("SiStripBaseDelayDummyDBWriter",
                                              record    = cms.string(""),
                                          OpenIovAt = cms.untracked.string("beginOfTime"),
                                          OpenIovAtTime = cms.untracked.uint32(1))
