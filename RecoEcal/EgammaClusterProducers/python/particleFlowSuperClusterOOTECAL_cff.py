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
from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy

run2_miniAOD_80XLegacy.toModify(
    particleFlowSuperClusterOOTECAL, 
    barrelRecHits = "reducedEcalRecHitsEB",
    endcapRecHits = "reducedEcalRecHitsEE"
)
run2_miniAOD_80XLegacy.toModify(
    particleFlowSuperClusterOOTECAL.regressionConfig, 
    ecalRecHitsEB = "reducedEcalRecHitsEB",
    ecalRecHitsEE = "reducedEcalRecHitsEE"
)

