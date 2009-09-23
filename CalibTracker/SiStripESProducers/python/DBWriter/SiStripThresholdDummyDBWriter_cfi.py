import FWCore.ParameterSet.Config as cms


siStripThresholdDummyDBWriter = cms.EDFilter("SiStripThresholdDummyDBWriter",
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1))




