import FWCore.ParameterSet.Config as cms

AODHIPhotonProducer = cms.EDProducer(
    "AODHIPhotonProducer",
    photonProducer = cms.InputTag("photons"),
    barrelEcalHits = cms.InputTag("ecalRecHit:EcalRecHitsEB"),
    endcapEcalHits = cms.InputTag("ecalRecHit:EcalRecHitsEE")
)
