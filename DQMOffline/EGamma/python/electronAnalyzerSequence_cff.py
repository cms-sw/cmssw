import FWCore.ParameterSet.Config as cms

mergedSuperClusters = cms.EDFilter("SuperClusterMerger",
    src = cms.VInputTag( 
      cms.InputTag("correctedHybridSuperClusters"),
      cms.InputTag("correctedMulti5x5SuperClustersWithPreshower")
    )
)

from DQMOffline.EGamma.electronAnalyzer_cff import *
dqmElectronAnalysis0 = dqmElectronAnalysis.clone() ;
dqmElectronAnalysis0.Selection = 0 ;
dqmElectronAnalysis1 = dqmElectronAnalysis.clone() ;
dqmElectronAnalysis1.Selection = 1 ;
dqmElectronAnalysis2 = dqmElectronAnalysis.clone() ;
dqmElectronAnalysis2.Selection = 2 ;
dqmElectronAnalysis3 = dqmElectronAnalysis.clone() ;
dqmElectronAnalysis3.Selection = 3 ;
dqmElectronAnalysis4 = dqmElectronAnalysis.clone() ;
dqmElectronAnalysis4.Selection = 4 ;

electronAnalyzerSequence = cms.Sequence(mergedSuperClusters*dqmElectronAnalysis0*dqmElectronAnalysis1*dqmElectronAnalysis2*dqmElectronAnalysis3*dqmElectronAnalysis4)
