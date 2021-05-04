from RecoEcal.EgammaClusterProducers.hybridSuperClusters_cfi import cleanedHybridSuperClusters
from RecoEcal.EgammaClusterProducers.multi5x5BasicClusters_cfi import multi5x5BasicClustersCleaned

from RecoEgamma.EgammaIsolationAlgos.electronTrackIsolations_cfi import trkIsol03CfgV1,trkIsol04CfgV1,trkIsol03CfgV2,trkIsol04CfgV2

from RecoEgamma.EgammaElectronProducers.gsfElectronProducer_cfi import gsfElectronProducer

#==============================================================================
# Producer of transient ecal driven gsf electrons
#==============================================================================

ecalDrivenGsfElectrons = gsfElectronProducer.clone(

    # Ecal rec hits configuration
    recHitFlagsToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitFlagToBeExcluded,
    recHitFlagsToBeExcludedEndcaps = multi5x5BasicClustersCleaned.RecHitFlagToBeExcluded,
    recHitSeverityToBeExcludedBarrel = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,
    recHitSeverityToBeExcludedEndcaps = cleanedHybridSuperClusters.RecHitSeverityToBeExcluded,

    # Isolation algos configuration
    trkIsol03Cfg = trkIsol03CfgV1,
    trkIsol04Cfg = trkIsol04CfgV1,
    trkIsolHEEP03Cfg = trkIsol03CfgV2,
    trkIsolHEEP04Cfg = trkIsol04CfgV2,
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(ecalDrivenGsfElectrons.preselection, minSCEtBarrel = 15.0)
pp_on_AA.toModify(ecalDrivenGsfElectrons.preselection, minSCEtEndcaps = 15.0)

ecalDrivenGsfElectronsFromMultiCl = ecalDrivenGsfElectrons.clone(
    gsfElectronCoresTag = "ecalDrivenGsfElectronCoresFromMultiCl",
    useGsfPfRecTracks = False,
    useDefaultEnergyCorrection = False,
    ambClustersOverlapStrategy = 0,
    applyAmbResolution = True,
    ignoreNotPreselected = False,
)
