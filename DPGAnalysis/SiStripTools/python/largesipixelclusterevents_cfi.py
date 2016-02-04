import FWCore.ParameterSet.Config as cms

largeSiPixelClusterEvents = cms.EDFilter('LargeSiPixelClusterEvents',
                                         collectionName = cms.InputTag("siPixelClusters"),
                                         absoluteThreshold = cms.untracked.int32(300),
                                         moduleThreshold = cms.untracked.int32(20),
                                         useQuality = cms.untracked.bool(False),
                                         qualityLabel = cms.untracked.string("")
)	
