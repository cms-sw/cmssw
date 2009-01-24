import FWCore.ParameterSet.Config as cms


siStripBadStripDummyDBWriter = cms.EDFilter("SiStripBadStripDummyDBWriter",
                                            OpenIovAt = cms.untracked.string("beginOfTime"),
                                            OpenIovAtTime = cms.untracked.uint32(1))




