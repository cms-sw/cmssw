import FWCore.ParameterSet.Config as cms


siStripBadStripFromQualityDummyDBWriter = cms.EDFilter("SiStripBadStripFromQualityDummyDBWriter",
                                                        OpenIovAt = cms.untracked.string("beginOfTime"),
                                                        OpenIovAtTime = cms.untracked.uint32(1))




