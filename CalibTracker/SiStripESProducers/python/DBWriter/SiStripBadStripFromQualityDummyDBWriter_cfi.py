import FWCore.ParameterSet.Config as cms


siStripBadStripFromQualityDummyDBWriter = cms.EDAnalyzer("SiStripBadStripFromQualityDummyDBWriter",
                                              record    = cms.string(""),
                                                        OpenIovAt = cms.untracked.string("beginOfTime"),
                                                        OpenIovAtTime = cms.untracked.uint32(1))




# foo bar baz
# v06itRrluq6Se
