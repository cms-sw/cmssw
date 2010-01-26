import FWCore.ParameterSet.Config as cms

LargeSiStripClusterEvents = cms.EDFilter('LargeSiStripClusterEvents',
                                         collectionName    = cms.InputTag("calZeroBiasClusters"),
                                         absoluteThreshold = cms.untracked.int32(10000),
                                         moduleThreshold   = cms.untracked.int32(20)
)	
