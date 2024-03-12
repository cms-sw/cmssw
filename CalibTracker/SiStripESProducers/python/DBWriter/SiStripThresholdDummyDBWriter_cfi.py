import FWCore.ParameterSet.Config as cms


siStripThresholdDummyDBWriter = cms.EDAnalyzer("SiStripThresholdDummyDBWriter",
                                              record    = cms.string(""),
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1))




# foo bar baz
# R7ffnse3kbgnm
# Lnta2EC3LXzy4
