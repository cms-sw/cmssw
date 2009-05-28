import FWCore.ParameterSet.Config as cms


siStripNoisesDummyDBWriter = cms.EDFilter("SiStripNoisesDummyDBWriter",
                                              record    = cms.string(""),
                                          OpenIovAt = cms.untracked.string("beginOfTime"),
                                          OpenIovAtTime = cms.untracked.uint32(1))




