import FWCore.ParameterSet.Config as cms


siStripDetVOffDummyDBWriter = cms.EDAnalyzer("SiStripDetVOffDummyDBWriter",
                                              record    = cms.string(""),
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1)
                                             )




# foo bar baz
# 3mA1T5MlJbqVK
# 7RcDbrfE4oLbK
