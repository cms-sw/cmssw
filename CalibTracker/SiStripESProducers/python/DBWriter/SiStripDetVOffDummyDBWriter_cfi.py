import FWCore.ParameterSet.Config as cms


siStripDetVOffDummyDBWriter = cms.EDAnalyzer("SiStripDetVOffDummyDBWriter",
                                              record    = cms.string(""),
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1)
                                             )




