import FWCore.ParameterSet.Config as cms

bysipixelvssistripclustmulteventfilter = cms.EDFilter('BySiPixelClusterVsSiStripClusterMultiplicityEventFilter',
                                                      multiplicityConfig = cms.PSet(
                                                                           firstMultiplicityConfig = cms.PSet(
                                                                                                     collectionName = cms.InputTag("siPixelClusters"),
                                                                                                     moduleThreshold = cms.untracked.int32(-1),
                                                                                                     useQuality = cms.untracked.bool(False),
                                                                                                     qualityLabel = cms.untracked.string("")
                                                                                                     ),
                                                                           secondMultiplicityConfig = cms.PSet(
                                                                                                      collectionName = cms.InputTag("siStripClusters"),
                                                                                                      moduleThreshold = cms.untracked.int32(-1),
                                                                                                      useQuality = cms.untracked.bool(False),
                                                                                                      qualityLabel = cms.untracked.string("")
                                                                                                      )
                                                                           ),
                                                      cut = cms.string("(mult2 > 30000) && ( mult2 > 20000+7*mult1)")
                                                      )
	
