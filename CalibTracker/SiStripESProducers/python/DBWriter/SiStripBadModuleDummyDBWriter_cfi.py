import FWCore.ParameterSet.Config as cms


siStripBadModuleDummyDBWriter = cms.EDFilter("SiStripBadModuleDummyDBWriter",
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1)
                                             )




