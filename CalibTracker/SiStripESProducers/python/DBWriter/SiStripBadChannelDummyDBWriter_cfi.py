import FWCore.ParameterSet.Config as cms


siStripBadChannelDummyDBWriter = cms.EDFilter("SiStripBadChannelDummyDBWriter",
                                              record    = cms.string(""),
                                              OpenIovAt = cms.untracked.string("beginOfTime"),
                                              OpenIovAtTime = cms.untracked.uint32(1))




