import FWCore.ParameterSet.Config as cms

photonIsolationHIProducer = cms.EDProducer(
    "photonIsolationHIProducer",
    photonProducer = cms.InputTag("photons"),
    ebRecHitCollection = cms.InputTag("ecalRecHit:EcalRecHitsEB"),
    eeRecHitCollection = cms.InputTag("ecalRecHit:EcalRecHitsEE"),
    hbhe = cms.InputTag("hbhereco"),
    hf = cms.InputTag("hfreco"),
    ho = cms.InputTag("horeco"),
    basicClusterBarrel = cms.InputTag("islandBasicClusters:islandBarrelBasicClusters"),
    basicClusterEndcap = cms.InputTag("islandBasicClusters:islandEndcapBasicClusters"),
    trackCollection = cms.InputTag("hiGeneralTracks"),
    trackQuality = cms.string("highPurity")
)

photonIsolationHIProducerpp = photonIsolationHIProducer.clone(
    trackCollection = "generalTracks"
)

photonIsolationHIProducerppGED = photonIsolationHIProducerpp.clone(
    photonProducer = "gedPhotons"
)

photonIsolationHIProducerppIsland = photonIsolationHIProducerpp.clone(
    photonProducer = "islandPhotons"
)

from RecoEcal.EgammaClusterProducers.islandBasicClusters_cfi import *

islandBasicClustersGED = islandBasicClusters.clone()
photonIsolationHITask = cms.Task(islandBasicClusters , photonIsolationHIProducerpp)
photonIsolationHITaskGED = cms.Task(islandBasicClustersGED , photonIsolationHIProducerppGED)
photonIsolationHITaskIsland = cms.Task(islandBasicClusters , photonIsolationHIProducerppIsland)
