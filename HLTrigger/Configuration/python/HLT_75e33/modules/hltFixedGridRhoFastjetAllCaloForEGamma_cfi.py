import FWCore.ParameterSet.Config as cms

hltFixedGridRhoFastjetAllCaloForEGamma = cms.EDProducer("FixedGridRhoProducerFastjetFromRecHit",
    gridSpacing = cms.double(0.55),
    maxRapidity = cms.double(2.5),
    hbheRecHitsTag = cms.InputTag( "hltHbhereco" ),
    ebRecHitsTag = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    eeRecHitsTag = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    skipHCAL = cms.bool( False ),
    skipECAL = cms.bool( False ),
    eThresHB = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
    eThresHE = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 )
)
