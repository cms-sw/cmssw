import FWCore.ParameterSet.Config as cms


siStripBadModuleDummyDBWriter = cms.EDAnalyzer("SiStripBadModuleDummyDBWriter",
                                              record    = cms.string(""),
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1)
                                             )




