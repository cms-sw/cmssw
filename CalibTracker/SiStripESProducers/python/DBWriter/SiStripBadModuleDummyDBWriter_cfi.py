import FWCore.ParameterSet.Config as cms


siStripBadModuleDummyDBWriter = cms.EDFilter("SiStripBadModuleDummyDBWriter",
                                              record    = cms.string(""),
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1)
                                             )




