import FWCore.ParameterSet.Config as cms

eidCutBasedExt = cms.EDProducer("EleIdCutBasedExtProducer",

    src = cms.InputTag("pixelMatchGsfElectrons"),
    reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    algorithm = cms.string('eIDCB'),

    electronQuality = cms.string('robust'),

    robustEleIDCuts = cms.PSet(
        barrel = cms.vdouble(0.115, 0.014, 0.09, 0.009),
        endcap = cms.vdouble(0.15, 0.0275, 0.092, 0.0105)
    ),
   
    tightEleIDCuts = cms.PSet(
        deltaPhiIn = cms.vdouble(0.032, 0.016, 0.0525, 0.09, 0.025, 0.035, 0.065, 0.092),
        hOverE = cms.vdouble(0.05, 0.042, 0.045, 0.0, 0.055, 0.037, 0.05, 0.0),
        sigmaEtaEta = cms.vdouble(0.0125, 0.011, 0.01, 0.0, 0.0265, 0.0252, 0.026, 0.0),
        deltaEtaIn = cms.vdouble(0.0055, 0.003, 0.0065, 0.0, 0.006, 0.0055, 0.0075, 0.0),
	eSeedOverPin = cms.vdouble(0.24, 0.94, 0.11, 0.0, 0.32, 0.83, 0.0, 0.0)
    ),
   
    looseEleIDCuts = cms.PSet(
        deltaPhiIn = cms.vdouble(0.05, 0.025, 0.053, 0.09, 0.07, 0.03, 0.092, 0.092),
        hOverE = cms.vdouble(0.115, 0.1, 0.055, 0.0, 0.145, 0.12, 0.15, 0.0),
        sigmaEtaEta = cms.vdouble(0.014, 0.012, 0.0115, 0.0, 0.0275, 0.0265, 0.0265, 0.0),
        deltaEtaIn = cms.vdouble(0.009, 0.0045, 0.0085, 0.0, 0.0105, 0.0068, 0.01, 0.0),
        eSeedOverPin = cms.vdouble(0.11, 0.91, 0.11, 0.0, 0.0, 0.85, 0.0, 0.0)
    )
   
   )


