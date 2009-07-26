import FWCore.ParameterSet.Config as cms

largeSiStripClusterEvents = cms.EDFilter('LargeSiStripDigiEvents',
                                         collectionName = cms.InputTag("siStripClusters"),
                                         absoluteThreshold = cms.untracked.int32(300),
                                         moduleThreshold = cms.untracked.int32(20)
)	
