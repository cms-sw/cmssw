import FWCore.ParameterSet.Config as cms

hltEgammaBcHcalIsolationProducersRegional= cms.EDProducer("EgammaHLTBcHcalIsolationProducersRegional",
                                                          recoEcalCandidateProducer = cms.InputTag("hltRecoEcalCandidate"),
                                                          caloTowerProducer         = cms.InputTag("hltTowerMakerForAll"),
                                                          rhoProducer = cms.InputTag("hltKT6CaloJets", "rho"),
                                                          doRhoCorrection           = cms.bool(False),
                                                          rhoScale                  = cms.double(1.),
                                                          rhoMax                    = cms.double(99999999.),
                                                          etMin                     = cms.double(-1.0),
                                                          innerCone                 = cms.double(0.0),
                                                          outerCone                 = cms.double(0.14),
                                                          depth                     = cms.int32(-1),
                                                          doEtSum                   = cms.bool(False),
                                                          effectiveAreaBarrel       = cms.double(0.021),
                                                          effectiveAreaEndcap       = cms.double(0.040)
                                                          )
