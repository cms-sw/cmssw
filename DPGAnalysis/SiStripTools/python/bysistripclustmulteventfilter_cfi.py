import FWCore.ParameterSet.Config as cms

bysistripclustmulteventfilter = cms.EDFilter('BySiStripClusterMultiplicityEventFilter',
                                             multiplicityConfig = cms.PSet(
                                                                  collectionName = cms.InputTag("siStripClusters"),
                                                                  moduleThreshold = cms.untracked.int32(20),
                                                                  useQuality = cms.untracked.bool(False),
                                                                  qualityLabel = cms.untracked.string("")
                                                                  ),
                                             cut = cms.string("mult > 300")
                                             )
	
