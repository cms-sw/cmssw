
import FWCore.ParameterSet.Config as cms

mergedSuperClusters = cms.EDProducer("SuperClusterMerger",
  src = cms.VInputTag( 
#    cms.InputTag("correctedHybridSuperClusters"),
#    cms.InputTag("correctedMulti5x5SuperClustersWithPreshower")
     cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
     cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower")
  )
)

from DQMOffline.EGamma.electronGeneralAnalyzer_cfi import *
dqmElectronGeneralAnalysis.OutputFolderName = cms.string("Egamma/Electrons/Ele1_General") ;

from DQMOffline.EGamma.electronAnalyzer_cfi import *
dqmElectronAnalysis.MinEt = cms.double(10.) ;
dqmElectronAnalysis.MaxTkIso03 = cms.double(1.) ;

dqmElectronAnalysisAllElectrons = dqmElectronAnalysis.clone(
    Selection = 0,
    OutputFolderName = "Egamma/Electrons/Ele2_All"
)
dqmElectronAnalysisSelectionEt = dqmElectronAnalysis.clone(
    Selection = 1,
    OutputFolderName = "Egamma/Electrons/Ele3_Et10"
)
dqmElectronAnalysisSelectionEtIso = dqmElectronAnalysis.clone(
    Selection = 2,
    OutputFolderName = "Egamma/Electrons/Ele4_Et10TkIso1"
)
#dqmElectronAnalysisSelectionEtIsoElID = dqmElectronAnalysis.clone() ;
#dqmElectronAnalysisSelectionEtIsoElID.Selection = 3 ;
#dqmElectronAnalysisSelectionEtIsoElID.OutputFolderName = cms.string("Egamma/Electrons/Ele4_Et10TkIso1ElID") ;

from DQMOffline.EGamma.electronTagProbeAnalyzer_cfi import *
dqmElectronTagProbeAnalysis.MinEt = cms.double(10.) ;
dqmElectronTagProbeAnalysis.MaxTkIso03 = cms.double(1.) ;
dqmElectronTagProbeAnalysis.OutputFolderName = cms.string("Egamma/Electrons/Ele5_TagAndProbe") ;

electronAnalyzerSequence = cms.Sequence(
   mergedSuperClusters
 * dqmElectronGeneralAnalysis
 * dqmElectronAnalysisAllElectrons
 * dqmElectronAnalysisSelectionEt
 * dqmElectronAnalysisSelectionEtIso
# * dqmElectronAnalysisSelectionEtIsoElID
 * dqmElectronTagProbeAnalysis
)

mergedSuperClustersHGC = mergedSuperClusters.clone(
    src = (
        ("particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel"),
        ("particleFlowSuperClusterHGCal")
    )
 )
dqmElectronAnalysisAllElectronsHGC = dqmElectronAnalysisAllElectrons.clone(
    OutputFolderName = 'Egamma/Electrons/Ele2HGC_All',
    MaxAbsEtaMatchingObject = 3.0,
    EtaMax = 3.0,
    EtaMin = -3.0,
    MaxAbsEta = 3.0,
    ElectronCollection = 'ecalDrivenGsfElectronsHGC',
    MatchingObjectCollection = 'mergedSuperClustersHGC'
)
_electronAnalyzerSequenceHGC = electronAnalyzerSequence.copy()
_electronAnalyzerSequenceHGC += cms.Sequence(mergedSuperClustersHGC+dqmElectronAnalysisAllElectronsHGC)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( mergedSuperClusters, src = cms.VInputTag( cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"), cms.InputTag("particleFlowSuperClusterHGCal","") ) )

phase2_hgcal.toReplaceWith(
electronAnalyzerSequence, _electronAnalyzerSequenceHGC
)


