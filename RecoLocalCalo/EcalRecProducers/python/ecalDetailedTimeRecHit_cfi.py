import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalTimeDigiParameters_cff import ecal_time_digi_parameters as digi_parameters

ecalDetailedTimeRecHit = cms.EDProducer("EcalDetailedTimeRecHitProducer",
                                        EKRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK"),
                                        EERecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                        EBRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                        EKTimeDigiCollection = cms.InputTag("mix","EKTimeDigi"),
                                        EETimeDigiCollection = cms.InputTag("mix","EETimeDigi"),
                                        EBTimeDigiCollection = cms.InputTag("mix","EBTimeDigi"),
                                        EBDetailedTimeRecHitCollection = cms.string('EcalRecHitsEB'),
                                        EEDetailedTimeRecHitCollection = cms.string('EcalRecHitsEE'),
                                        EKDetailedTimeRecHitCollection = cms.string('EcalRecHitsEK'),
                                        EBTimeLayer = digi_parameters.timeLayerBarrel,
                                        EETimeLayer = digi_parameters.timeLayerEndcap,
                                        EKTimeLayer = digi_parameters.timeLayerShashlik,
                                        correctForVertexZPosition=cms.bool(True),
                                        useMCTruthVertex=cms.bool(True),
                                        recoVertex = cms.InputTag("offlinePrimaryVerticesWithBS"),
                                        simVertex = cms.InputTag("g4SimHits")
                                        )
