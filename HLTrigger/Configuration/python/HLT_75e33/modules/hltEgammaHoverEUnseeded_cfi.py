import FWCore.ParameterSet.Config as cms

hltEgammaHoverEUnseeded = cms.EDProducer("EgammaHLTHcalVarProducerFromRecHit",
    absEtaLowEdges = cms.vdouble(0.0, 1.479),
    depth = cms.int32(0),
    doEtSum = cms.bool(False),
    doRhoCorrection = cms.bool(False),
    effectiveAreas = cms.vdouble(0.105, 0.17),
    innerCone = cms.double(0.0),
    outerCone = cms.double(0.14),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesUnseeded"),
    rhoMax = cms.double(99999999.0),
    rhoProducer = cms.InputTag("hltFixedGridRhoFastjetAllCaloForEGamma"),
    rhoScale = cms.double(1.0),
    useSingleTower = cms.bool(False),
    hbheRecHitsTag = cms.InputTag( "hltHbhereco" ),
    eThresHB = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ), #Run3 thresholds. Will be overwritten with valid aging customisation
    etThresHB = cms.vdouble( 0.0, 0.0, 0.0, 0.0 ),
    eThresHE = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
    etThresHE = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
    usePFThresholdsFromDB = cms.bool(True),
    maxSeverityHB = cms.int32( 9 ),
    maxSeverityHE = cms.int32( 9 )
)
