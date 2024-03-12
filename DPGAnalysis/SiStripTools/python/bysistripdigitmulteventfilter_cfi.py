import FWCore.ParameterSet.Config as cms

bysistripdigimulteventfilter = cms.EDFilter('BySiStripDigiMultiplicityEventFilter',
                                            multiplicityConfig = cms.PSet(
                                                               collectionName = cms.InputTag("siStripDigis","ZeroSuppressed"),
                                                               moduleThreshold = cms.untracked.int32(-1)
                                                               ),
                                            cut = cms.string("mult > 100000")
                                            )	
# foo bar baz
# 89jg30ZUNYN0m
# b7xY6P4qlxmF7
