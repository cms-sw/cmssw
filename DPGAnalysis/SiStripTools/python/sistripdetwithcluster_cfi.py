import FWCore.ParameterSet.Config as cms

sistripdetwithcluster = cms.EDAnalyzer('SiStripDetWithCluster',
                                       collectionName = cms.InputTag("siStripClusters"),     
                                       selectedModules = cms.untracked.vuint32()
)	
