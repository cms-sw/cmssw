import FWCore.ParameterSet.Config as cms


siStripFedCablingDummyDBWriter = cms.EDAnalyzer("SiStripFedCablingDummyDBWriter",
                                              record    = cms.string(""),
                                              OpenIovAt = cms.untracked.string("beginOfTime"),
                                              OpenIovAtTime = cms.untracked.uint32(1))




