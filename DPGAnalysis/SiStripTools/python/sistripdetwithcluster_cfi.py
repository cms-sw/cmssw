import FWCore.ParameterSet.Config as cms

sistripdetwithcluster = cms.EDFilter('SiStripDetWithCluster',
                                     collectionName = cms.InputTag("siStripClusters"),     
                                     selectedModules = cms.untracked.vuint32()
)	
# foo bar baz
# x7X1np0Kc7gg2
