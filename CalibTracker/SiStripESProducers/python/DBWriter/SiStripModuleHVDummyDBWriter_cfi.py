import FWCore.ParameterSet.Config as cms


siStripModuleHVDummyDBWriter = cms.EDFilter("SiStripModuleHVDummyDBWriter",
                                              record    = cms.string(""),
                                             OpenIovAt = cms.untracked.string("beginOfTime"),
                                             OpenIovAtTime = cms.untracked.uint32(1)
                                             )




