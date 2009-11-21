import FWCore.ParameterSet.Config as cms


siStripConfObjectDummyDBWriter = cms.EDFilter("SiStripConfObjectDummyDBWriter",
                                              record    = cms.string(""),
                                          OpenIovAt = cms.untracked.string("beginOfTime"),
                                          OpenIovAtTime = cms.untracked.uint32(1))




