import FWCore.ParameterSet.Config as cms

largeSiStripClusterEvents = cms.EDFilter('LargeSiStripClusterEvents',
                                         collectionName = cms.InputTag("siStripClusters"),
                                         absoluteThreshold = cms.untracked.int32(300),
                                         moduleThreshold = cms.untracked.int32(20),
                                         useQuality = cms.untracked.bool(False),
                                         qualityLabel = cms.untracked.string("")
)	
# foo bar baz
# 1dp61JjnooNA7
# poMLRGJjifMG4
