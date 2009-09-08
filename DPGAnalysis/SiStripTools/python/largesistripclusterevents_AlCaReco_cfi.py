import FWCore.ParameterSet.Config as cms

LargeSiStripClusterEvents = cms.EDFilter('LargeSiStripClusterEvents',
                                         collectionName    = cms.InputTag("calZeroBiasClusters"),
                                         absoluteThreshold = cms.untracked.int32(300),
                                         moduleThreshold   = cms.untracked.int32(20)
)	
