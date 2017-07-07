import FWCore.ParameterSet.Config as cms
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterECAL_cfi import *

particleFlowSuperClusterOOTECAL = particleFlowSuperClusterECAL.clone()
particleFlowSuperClusterOOTECAL.PFClusters = cms.InputTag("particleFlowClusterOOTECAL")
particleFlowSuperClusterOOTECAL.ESAssociation = cms.InputTag("particleFlowClusterOOTECAL")
particleFlowSuperClusterOOTECAL.PFBasicClusterCollectionBarrel = cms.string("particleFlowBasicClusterOOTECALBarrel")
particleFlowSuperClusterOOTECAL.PFSuperClusterCollectionBarrel = cms.string("particleFlowSuperClusterOOTECALBarrel")
particleFlowSuperClusterOOTECAL.PFBasicClusterCollectionEndcap = cms.string("particleFlowBasicClusterOOTECALEndcap")
particleFlowSuperClusterOOTECAL.PFSuperClusterCollectionEndcap = cms.string("particleFlowSuperClusterOOTECALEndcap")
particleFlowSuperClusterOOTECAL.PFBasicClusterCollectionPreshower = cms.string("particleFlowBasicClusterOOTECALPreshower")
particleFlowSuperClusterOOTECAL.PFSuperClusterCollectionEndcapWithPreshower = cms.string("particleFlowSuperClusterOOTECALEndcapWithPreshower")

## modification for Algo
particleFlowSuperClusterOOTECAL.isOOTCollection = cms.bool(True)
particleFlowSuperClusterOOTECAL.barrelRecHits = cms.InputTag("ecalRecHit","EcalRecHitsEB")
particleFlowSuperClusterOOTECAL.endcapRecHits = cms.InputTag("ecalRecHit","EcalRecHitsEE")
