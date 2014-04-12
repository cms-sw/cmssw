import FWCore.ParameterSet.Config as cms


siStripBadFiberDummyDBWriter = cms.EDAnalyzer("SiStripBadFiberDummyDBWriter",
                                              record    = cms.string(""),
                                              OpenIovAt = cms.untracked.string("beginOfTime"),
                                              OpenIovAtTime = cms.untracked.uint32(1))




