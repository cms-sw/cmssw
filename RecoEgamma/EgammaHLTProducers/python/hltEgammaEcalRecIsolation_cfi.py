import FWCore.ParameterSet.Config as cms

hltEgammaEcalRecIsolationProducer= cms.EDProducer( "EgammaHLTEcalRecIsolationProducer",
                                                   recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
                                                   ecalBarrelRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
                                                   ecalBarrelRecHitCollection = cms.InputTag( "EcalRecHitsEB" ),
                                                   ecalEndcapRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
                                                   ecalEndcapRecHitCollection = cms.InputTag( "EcalRecHitsEE" ),
                                                   rhoProducer = cms.InputTag("hltKT6CaloJets", "rho"),
                                                   doRhoCorrection           = cms.bool(False),
                                                   rhoScale                  = cms.double(1.),
                                                   rhoMax                    = cms.double(99999999.),
                                                   intRadiusBarrel = cms.double( 0.045 ),
                                                   intRadiusEndcap = cms.double( 0.07 ),
                                                   extRadius = cms.double( 0.4 ),
                                                   etMinBarrel = cms.double( -9999.0 ),
                                                   eMinBarrel = cms.double( 0.08 ),
                                                   etMinEndcap = cms.double( -9999.0 ),
                                                   eMinEndcap = cms.double( 0.3 ),
                                                   jurassicWidth = cms.double( 0.02 ),
                                                   useIsolEt = cms.bool( True ),
                                                   tryBoth = cms.bool( True ),
                                                   subtract = cms.bool( False ),
                                                   useNumCrystals = cms.bool( False ),
                                                   effectiveAreaBarrel       = cms.double(0.101),
                                                   effectiveAreaEndcap       = cms.double(0.046)
                                                   )

