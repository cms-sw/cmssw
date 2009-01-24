import FWCore.ParameterSet.Config as cms


siStripBadFiberDummyDBWriter = cms.EDFilter("SiStripBadFiberDummyDBWriter",
                                            OpenIovAt = cms.untracked.string("beginOfTime"),
                                            OpenIovAtTime = cms.untracked.uint32(1))




