import FWCore.ParameterSet.Config as cms


siStripLorentzAngleDummyDBWriter = cms.EDFilter("SiStripLorentzAngleDummyDBWriter",
                                                OpenIovAt = cms.untracked.string("beginOfTime"),
                                                OpenIovAtTime = cms.untracked.uint32(1))




