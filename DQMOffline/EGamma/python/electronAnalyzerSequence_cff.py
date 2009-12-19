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
dqmElectronAnalysisAllElectrons.OutputFolderName = cms.string("AllElectrons") ;
dqmElectronAnalysisSelectionEt = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisSelectionEt.Selection = 1 ;
dqmElectronAnalysisSelectionEt.OutputFolderName = cms.string("Et10") ;
dqmElectronAnalysisSelectionEtIso = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisSelectionEtIso.Selection = 2 ;
dqmElectronAnalysisSelectionEtIso.OutputFolderName = cms.string("Et10Iso5") ;
dqmElectronAnalysisSelectionEtIsoElID = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisSelectionEtIsoElID.Selection = 3 ;
dqmElectronAnalysisSelectionEtIsoElID.OutputFolderName = cms.string("Et10Iso5ElID") ;
dqmElectronAnalysisTagAndProbe = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisTagAndProbe.Selection = 4 ;
dqmElectronAnalysisTagAndProbe.OutputFolderName = cms.string("TagAndProbe") ;

electronAnalyzerSequence = cms.Sequence(
   mergedSuperClusters
 * dqmElectronAnalysisAllElectrons
 * dqmElectronAnalysisSelectionEt
 * dqmElectronAnalysisSelectionEtIso
# * dqmElectronAnalysisSelectionEtIsoElID
# * dqmElectronAnalysisTagAndProbe
)
