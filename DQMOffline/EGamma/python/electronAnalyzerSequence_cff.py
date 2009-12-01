import FWCore.ParameterSet.Config as cms

mergedSuperClusters = cms.EDFilter("SuperClusterMerger",
    src = cms.VInputTag( 
      cms.InputTag("correctedHybridSuperClusters"),
      cms.InputTag("correctedMulti5x5SuperClustersWithPreshower")
    )
)

from DQMOffline.EGamma.electronAnalyzer_cff import *
dqmElectronAnalysisAllElectrons = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisAllElectrons.Selection = 0 ;
dqmElectronAnalysisSelectionEt = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisSelectionEt.Selection = 1 ;
dqmElectronAnalysisSelectionEtIso = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisSelectionEtIso.Selection = 2 ;
dqmElectronAnalysisSelectionEtIsoElID = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisSelectionEtIsoElID.Selection = 3 ;
dqmElectronAnalysisTagAndProbe = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisTagAndProbe.Selection = 4 ;

electronAnalyzerSequence = cms.Sequence(
   mergedSuperClusters
 * dqmElectronAnalysisAllElectrons
 * dqmElectronAnalysisSelectionEt
 * dqmElectronAnalysisSelectionEtIso
 * dqmElectronAnalysisSelectionEtIsoElID
 * dqmElectronAnalysisTagAndProbe
)
