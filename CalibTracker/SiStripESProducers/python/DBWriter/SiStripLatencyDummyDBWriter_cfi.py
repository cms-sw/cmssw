import FWCore.ParameterSet.Config as cms


siStripLatencyDummyDBWriter = cms.EDAnalyzer("SiStripLatencyDummyDBWriter",
                                              record    = cms.string(""),
                                          OpenIovAt = cms.untracked.string("beginOfTime"),
                                          OpenIovAtTime = cms.untracked.uint32(1))




