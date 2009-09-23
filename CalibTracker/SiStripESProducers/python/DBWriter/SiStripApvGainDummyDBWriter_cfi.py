import FWCore.ParameterSet.Config as cms


siStripApvGainDummyDBWriter = cms.EDFilter("SiStripApvGainDummyDBWriter",
                                           OpenIovAt = cms.untracked.string("beginOfTime"),
                                           OpenIovAtTime = cms.untracked.uint32(1))




