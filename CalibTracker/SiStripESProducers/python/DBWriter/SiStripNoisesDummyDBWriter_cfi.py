import FWCore.ParameterSet.Config as cms


siStripNoisesDummyDBWriter = cms.EDAnalyzer("SiStripNoisesDummyDBWriter",
                                              record    = cms.string(""),
                                          OpenIovAt = cms.untracked.string("beginOfTime"),
                                          OpenIovAtTime = cms.untracked.uint32(1))




