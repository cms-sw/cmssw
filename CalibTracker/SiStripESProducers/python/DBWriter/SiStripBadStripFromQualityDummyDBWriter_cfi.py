import FWCore.ParameterSet.Config as cms


siStripBadStripFromQualityDummyDBWriter = cms.EDFilter("SiStripBadStripFromQualityDummyDBWriter",
                                              record    = cms.string(""),
                                                        OpenIovAt = cms.untracked.string("beginOfTime"),
                                                        OpenIovAtTime = cms.untracked.uint32(1))




