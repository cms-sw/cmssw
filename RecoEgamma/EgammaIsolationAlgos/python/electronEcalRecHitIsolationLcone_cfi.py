import FWCore.ParameterSet.Config as cms

electronEcalRecHitIsolationLcone = cms.EDProducer("EgammaEcalRecHitIsolationProducer",

    ecalBarrelRecHitProducer = cms.InputTag("ecalRecHit"),
    ecalBarrelRecHitCollection = cms.InputTag("EcalRecHitsEB"),
    ecalEndcapRecHitProducer = cms.InputTag("ecalRecHit"),
    ecalEndcapRecHitCollection = cms.InputTag("EcalRecHitsEE"),

    #useNumCrystals = cms.bool(False),
    #intRadiusBarrel = cms.double(0.045),
    #intRadiusEndcap = cms.double(0.070),
    #jurassicWidth = cms.double(0.02),    #dEta strip width
    useNumCrystals = cms.bool(True),
    intRadiusBarrel = cms.double(3.0),
    intRadiusEndcap = cms.double(3.0),
    jurassicWidth = cms.double(1.5),    #dEta strip width
    extRadius = cms.double(0.4),
    etMinBarrel = cms.double(0.0),
    eMinBarrel = cms.double(0.095),
    etMinEndcap = cms.double(0.110),
    eMinEndcap = cms.double(0.0),

    useIsolEt = cms.bool(True),
    tryBoth   = cms.bool(True),
    subtract  = cms.bool(False),
    vetoClustered  = cms.bool(False),

    emObjectProducer = cms.InputTag("gedGsfElectrons")
)


