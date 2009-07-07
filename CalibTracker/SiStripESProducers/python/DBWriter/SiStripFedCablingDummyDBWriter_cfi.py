import FWCore.ParameterSet.Config as cms


siStripFedCablingDummyDBWriter = cms.EDFilter("SiStripFedCablingDummyDBWriter",
                                              OpenIovAt = cms.untracked.string("beginOfTime"),
                                              OpenIovAtTime = cms.untracked.uint32(1))




