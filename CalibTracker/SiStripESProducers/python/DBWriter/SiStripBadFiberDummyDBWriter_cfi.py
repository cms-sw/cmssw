import FWCore.ParameterSet.Config as cms


siStripBadFiberDummyDBWriter = cms.EDFilter("SiStripBadFiberDummyDBWriter",
                                              record    = cms.string(""),
                                            OpenIovAt = cms.untracked.string("beginOfTime"),
                                            OpenIovAtTime = cms.untracked.uint32(1))




