import FWCore.ParameterSet.Config as cms


siStripBadChannelDummyDBWriter = cms.EDFilter("SiStripBadChannelDummyDBWriter",
                                              OpenIovAt = cms.untracked.string("beginOfTime"),
                                              OpenIovAtTime = cms.untracked.uint32(1))




