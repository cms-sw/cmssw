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
