import FWCore.ParameterSet.Config as cms

bysipixelclustmulteventfilter = cms.EDFilter('BySiPixelClusterMultiplicityEventFilter',
                                             multiplicityConfig = cms.PSet(
                                                                  collectionName = cms.InputTag("siPixelClusters"),
                                                                  moduleThreshold = cms.untracked.int32(20),
                                                                  useQuality = cms.untracked.bool(False),
                                                                  qualityLabel = cms.untracked.string("")
                                                                  ),
                                             cut = cms.string("mult > 300")
                                             )
	
