import FWCore.ParameterSet.Config as cms

#  AlCaElectrons producer
alCaIsolatedElectrons = cms.EDProducer("AlCaElectronsProducer",
    electronLabel = cms.InputTag("electronFilter"),
    alcaEndcapHitCollection = cms.string('alcaEndcapHits'),
    phiSize = cms.int32(11),
    etaSize = cms.int32(5),
    ebRecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    alcaBarrelHitCollection = cms.string('alcaBarrelHits'),
    eeRecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)


