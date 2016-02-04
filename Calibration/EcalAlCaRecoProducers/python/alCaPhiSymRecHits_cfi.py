import FWCore.ParameterSet.Config as cms

# AlCaPhiSymRecHits producer
alCaPhiSymRecHits = cms.EDProducer("AlCaPhiSymRecHitsProducer",
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    eCut_barrel = cms.double(0.15),
    ecalRecHitsProducer = cms.string('ecalRecHit'),
    eCut_endcap = cms.double(0.75),
    VerbosityLevel = cms.string('ERROR'),
    phiSymBarrelHitCollection = cms.string('phiSymEcalRecHitsEB'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    phiSymEndcapHitCollection = cms.string('phiSymEcalRecHitsEE')
)


