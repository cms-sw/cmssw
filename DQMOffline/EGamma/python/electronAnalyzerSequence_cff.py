import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.electronAnalyzer_cff import *
mergedSuperClusters = cms.EDFilter("SuperClusterMerger",
    src = cms.VInputTag(cms.InputTag("correctedHybridSuperClusters"), cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"))
)

electronAnalyzerSequence = cms.Sequence(mergedSuperClusters*gsfElectronAnalysis)
