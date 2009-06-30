import FWCore.ParameterSet.Config as cms


siStripModuleHVDummyDBWriter = cms.EDFilter("SiStripModuleHVDummyDBWriter",
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1)
                                             )




