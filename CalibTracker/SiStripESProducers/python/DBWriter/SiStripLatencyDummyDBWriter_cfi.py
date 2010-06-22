import FWCore.ParameterSet.Config as cms


siStripLatencyDummyDBWriter = cms.EDFilter("SiStripLatencyDummyDBWriter",
                                              record    = cms.string(""),
                                          OpenIovAt = cms.untracked.string("beginOfTime"),
                                          OpenIovAtTime = cms.untracked.uint32(1))




