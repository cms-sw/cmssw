import FWCore.ParameterSet.Config as cms


siStripBackPlaneCorrectionDummyDBWriter = cms.EDAnalyzer("SiStripBackPlaneCorrectionDummyDBWriter",
                                              record    = cms.string(""),
                                                OpenIovAt = cms.untracked.string("beginOfTime"),
                                                OpenIovAtTime = cms.untracked.uint32(1))




