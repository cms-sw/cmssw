import FWCore.ParameterSet.Config as cms


siStripDetVOffDummyDBWriter = cms.EDFilter("SiStripDetVOffDummyDBWriter",
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1)
                                             )




