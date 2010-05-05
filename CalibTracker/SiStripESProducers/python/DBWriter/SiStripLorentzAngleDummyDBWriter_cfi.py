import FWCore.ParameterSet.Config as cms


siStripLorentzAngleDummyDBWriter = cms.EDAnalyzer("SiStripLorentzAngleDummyDBWriter",
                                              record    = cms.string(""),
                                                OpenIovAt = cms.untracked.string("beginOfTime"),
                                                OpenIovAtTime = cms.untracked.uint32(1))




