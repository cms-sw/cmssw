import FWCore.ParameterSet.Config as cms


siStripPedestalsDummyDBWriter = cms.EDFilter("SiStripPedestalsDummyDBWriter",
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1))




