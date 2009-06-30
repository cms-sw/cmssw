import FWCore.ParameterSet.Config as cms


siStripNoisesDummyDBWriter = cms.EDFilter("SiStripNoisesDummyDBWriter",
                                          OpenIovAt = cms.untracked.string("beginOfTime"),
                                          OpenIovAtTime = cms.untracked.uint32(1))




