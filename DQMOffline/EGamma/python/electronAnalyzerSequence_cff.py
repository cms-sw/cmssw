import FWCore.ParameterSet.Config as cms

mergedSuperClusters = cms.EDFilter("SuperClusterMerger",
    src = cms.VInputTag( 
      cms.InputTag("correctedHybridSuperClusters"),
      cms.InputTag("correctedMulti5x5SuperClustersWithPreshower")
    )
)

from DQMOffline.EGamma.electronGeneralAnalyzer_cfi import *

from DQMOffline.EGamma.electronAnalyzer_cfi import *
dqmElectronAnalysis.MinEt = cms.double(10.) ;
dqmElectronAnalysis.MaxTkIso03 = cms.double(1.) ;

dqmElectronAnalysisAllElectrons = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisAllElectrons.Selection = 0 ;
dqmElectronAnalysisAllElectrons.OutputFolderName = cms.string("AllElectrons") ;

dqmElectronAnalysisSelectionEt = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisSelectionEt.Selection = 1 ;
dqmElectronAnalysisSelectionEt.OutputFolderName = cms.string("Et10") ;

dqmElectronAnalysisSelectionEtIso = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisSelectionEtIso.Selection = 2 ;
dqmElectronAnalysisSelectionEtIso.OutputFolderName = cms.string("Et10Iso1") ;

dqmElectronAnalysisSelectionEtIsoElID = dqmElectronAnalysis.clone() ;
dqmElectronAnalysisSelectionEtIsoElID.Selection = 3 ;
dqmElectronAnalysisSelectionEtIsoElID.OutputFolderName = cms.string("Et10Iso1ElID") ;

from DQMOffline.EGamma.electronTagProbeAnalyzer_cfi import *
dqmElectronTagProbeAnalysis.MinEt = cms.double(10.) ;
dqmElectronTagProbeAnalysis.MaxTkIso03 = cms.double(1.) ;

electronAnalyzerSequence = cms.Sequence(
   mergedSuperClusters
 * dqmElectronGeneralAnalysis
 * dqmElectronAnalysisAllElectrons
 * dqmElectronAnalysisSelectionEt
 * dqmElectronAnalysisSelectionEtIso
# * dqmElectronAnalysisSelectionEtIsoElID
 * dqmElectronTagProbeAnalysis
)
