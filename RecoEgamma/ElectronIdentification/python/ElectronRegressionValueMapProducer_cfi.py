import FWCore.ParameterSet.Config as cms

electronRegressionValueMapProducer = cms.EDProducer('ElectronRegressionValueMapProducer',
                                                    #presently the electron regressions use the "zero-suppressed" shower shapes
                                                    useFull5x5 = cms.bool(False), 
                                                    # The module automatically detects AOD vs miniAOD, so we configure both
                                                    #
                                                    # AOD case
                                                    #
                                                    ebReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEB"),
                                                    eeReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsEE"),
                                                    esReducedRecHitCollection = cms.InputTag("reducedEcalRecHitsES"),
                                                    src = cms.InputTag('gedGsfElectrons'),
                                                    #
                                                    # miniAOD case
                                                    #
                                                    ebReducedRecHitCollectionMiniAOD = cms.InputTag("reducedEgamma:reducedEBRecHits"),
                                                    eeReducedRecHitCollectionMiniAOD = cms.InputTag("reducedEgamma:reducedEERecHits"),
                                                    esReducedRecHitCollectionMiniAOD = cms.InputTag("reducedEgamma:reducedESRecHits"),
                                                    srcMiniAOD = cms.InputTag('slimmedElectrons',processName=cms.InputTag.skipCurrentProcess()),
                                                    )
