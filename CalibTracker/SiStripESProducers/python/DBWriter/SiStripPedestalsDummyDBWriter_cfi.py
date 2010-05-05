import FWCore.ParameterSet.Config as cms


siStripPedestalsDummyDBWriter = cms.EDAnalyzer("SiStripPedestalsDummyDBWriter",
                                              record    = cms.string(""),
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1))




