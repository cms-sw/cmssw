import FWCore.ParameterSet.Config as cms

hltEgammaHcalIsolationProducersRegional= cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
                                                         recoEcalCandidateProducer = cms.InputTag( "hltRecoEcalCandidate" ),
                                                         hbheRecHitProducer = cms.InputTag( "hbhereco" ),
                                                         rhoProducer = cms.InputTag("hltKT6CaloJets", "rho"),
                                                         doRhoCorrection           = cms.bool(False),
                                                         rhoScale                  = cms.double(1.),
                                                         rhoMax                    = cms.double(99999999.),
                                                         eMinHB = cms.double( 0.7 ),
                                                         eMinHE = cms.double( 0.8 ),
                                                         etMinHB = cms.double( -1.0 ),
                                                         etMinHE = cms.double( -1.0 ),
                                                         innerCone = cms.double( 0.0 ),
                                                         outerCone = cms.double( 0.14 ),
                                                         depth = cms.int32( -1 ),
                                                         doEtSum = cms.bool( False ),
                                                         effectiveAreaBarrel       = cms.double(0.021),
                                                         effectiveAreaEndcap       = cms.double(0.040)             
                                                         )

