import FWCore.ParameterSet.Config as cms


siStripBadStripDummyDBWriter = cms.EDAnalyzer("SiStripBadStripDummyDBWriter",
                                              record    = cms.string(""),
                                            OpenIovAt = cms.untracked.string("beginOfTime"),
                                            OpenIovAtTime = cms.untracked.uint32(1))




# foo bar baz
# 1lo8UrfohaMdW
# 2LTjpA6bsjC2P
