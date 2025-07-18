import FWCore.ParameterSet.Config as cms
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECAL_cfi import *

particleFlowSuperClusterOOTECAL = particleFlowSuperClusterECAL.clone(
    PFClusters                     = "particleFlowClusterOOTECAL",
    ESAssociation                  = "particleFlowClusterOOTECAL",
    PFBasicClusterCollectionBarrel = "particleFlowBasicClusterOOTECALBarrel",
    PFSuperClusterCollectionBarrel = "particleFlowSuperClusterOOTECALBarrel",
    PFBasicClusterCollectionEndcap = "particleFlowBasicClusterOOTECALEndcap",
    PFSuperClusterCollectionEndcap = "particleFlowSuperClusterOOTECALEndcap",
    PFBasicClusterCollectionPreshower = "particleFlowBasicClusterOOTECALPreshower",
    PFSuperClusterCollectionEndcapWithPreshower = "particleFlowSuperClusterOOTECALEndcapWithPreshower",
    ## modification for Algo
    isOOTCollection                = True,
    barrelRecHits                  = "ecalRecHit:EcalRecHitsEB",
    endcapRecHits                  = "ecalRecHit:EcalRecHitsEE"
)
