import FWCore.ParameterSet.Config as cms

largeSiStripClusterEvents = cms.EDFilter('LargeSiStripClusterEvents',
                                         collectionName = cms.InputTag("siStripClusters"),
                                         absoluteThreshold = cms.untracked.int32(300),
                                         moduleThreshold = cms.untracked.int32(20)
)	
