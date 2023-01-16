import FWCore.ParameterSet.Config as cms

hltEgammaHoverEL1Seeded = cms.EDProducer("EgammaHLTHcalVarProducerFromRecHit",
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    depth = cms.int32(0),
    doEtSum = cms.bool(False),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.105, 0.17),
    innerCone = cms.double(0.0),
    outerCone = cms.double(0.14),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    rhoMax = cms.double(99999999.0),
    rhoProducer = cms.InputTag("hltFixedGridRhoFastjetAllCaloForEGamma"),
    rhoScale = cms.double(1.0),
    useSingleTower = cms.bool(False),
    hbheRecHitsTag = cms.InputTag( "hltHbhereco" ),
    eThresHB = cms.vdouble( 0.8, 1.2, 1.2, 1.2 ),
    etThresHB = cms.vdouble( 0.0, 0.0, 0.0, 0.0 ),
    eThresHE = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
    etThresHE = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
    maxSeverityHB = cms.int32( 9 ),
    maxSeverityHE = cms.int32( 9 )
)
