import FWCore.ParameterSet.Config as cms


siStripBadChannelDummyDBWriter = cms.EDAnalyzer("SiStripBadChannelDummyDBWriter",
                                                record    = cms.string(""),
                                                OpenIovAt = cms.untracked.string("beginOfTime"),
                                                OpenIovAtTime = cms.untracked.uint32(1))




