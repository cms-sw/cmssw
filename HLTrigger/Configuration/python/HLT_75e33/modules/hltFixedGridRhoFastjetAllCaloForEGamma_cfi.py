import FWCore.ParameterSet.Config as cms

hltFixedGridRhoFastjetAllCaloForEGamma = cms.EDProducer("FixedGridRhoProducerFastjetFromRecHit",
    eThresHB = cms.vdouble(0.8, 1.2, 1.2, 1.2),
    eThresHE = cms.vdouble(
        0.1, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.2
    ),
    ebRecHitsTag = cms.InputTag("hltEcalRecHit","EcalRecHitsEB"),
    eeRecHitsTag = cms.InputTag("hltEcalRecHit","EcalRecHitsEE"),
    gridSpacing = cms.double(0.55),
    hbheRecHitsTag = cms.InputTag("hltHbhereco"),
    maxRapidity = cms.double(2.5),
    skipECAL = cms.bool(False),
    skipHCAL = cms.bool(False)
)
