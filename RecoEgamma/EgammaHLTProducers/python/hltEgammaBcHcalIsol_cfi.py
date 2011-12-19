import FWCore.ParameterSet.Config as cms

hltEgammaHcalIsolationProducersRegional= cms.EDProducer("EgammaHLTBcHcalIsolationProducersRegional",
                                                        recoEcalCandidateProducer = cms.InputTag("hltRecoEcalCandidate"),
                                                        caloTowerProducer         = cms.InputTag("hltTowerMakerForAll"),
                                                        etMin                     = cms.double(-1.0),
                                                        innerCone                 = cms.double(0.0),
                                                        outerCone                 = cms.double(0.14),
                                                        depth                     = cms.int32(-1),
                                                        doEtSum                   = cms.bool(False),
                                                        effectiveAreaBarrel       = cms.double(0.021),
                                                        effectiveAreaEndcap       = cms.double(0.040)
                                                        )
