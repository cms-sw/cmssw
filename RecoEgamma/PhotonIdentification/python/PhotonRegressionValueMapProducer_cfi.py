import FWCore.ParameterSet.Config as cms

photonRegressionValueMapProducer = cms.EDProducer('PhotonRegressionValueMapProducer',
                                                  #presently the photon regressions use the fraction-ized (PF clustering) shower shapes
                                                  useFull5x5 = cms.bool(False), 
                                                  # The module automatically detects AOD vs miniAOD, so we configure both
                                                  #
                                                  # AOD case
                                                  #
                                                  ebReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                                                  eeReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                                                  esReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsES"),
                                                  src = cms.InputTag('gedPhotons'),
                                                  #
                                                  # miniAOD case
                                                  #
                                                  ebReducedRecHitCollectionMiniAOD = cms.InputTag("reducedEgamma:reducedEBRecHits"),
                                                  eeReducedRecHitCollectionMiniAOD = cms.InputTag("reducedEgamma:reducedEERecHits"),
                                                  esReducedRecHitCollectionMiniAOD = cms.InputTag("reducedEgamma:reducedESRecHits"),
                                                  srcMiniAOD = cms.InputTag('slimmedPhotons',
                                                                            processName=cms.InputTag.skipCurrentProcess()),
                                                  )
