import FWCore.ParameterSet.Config as cms

largeSiStripDigiEvents = cms.EDFilter('LargeSiStripDigiEvents',
                                      collectionName = cms.InputTag("siStripDigis","ZeroSuppressed"),
                                      absoluteThreshold = cms.untracked.int32(100000),
                                      moduleThreshold = cms.untracked.int32(-1)
)	
# foo bar baz
# qfETLaZSsPE76
# fP4BzA9Zfl9Av
