import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalTimeDigiParameters_cff import ecal_time_digi_parameters as digi_parameters

ecalDetailedTimeRecHit = cms.EDProducer("EcalDetailedTimeRecHitProducer",
                                        EERecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                        EBRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                        EETimeDigiCollection = cms.InputTag("mix","EETimeDigi"),
                                        EBTimeDigiCollection = cms.InputTag("mix","EBTimeDigi"),
                                        EBDetailedTimeRecHitCollection = cms.string('EcalRecHitsEB'),
                                        EEDetailedTimeRecHitCollection = cms.string('EcalRecHitsEE'),
                                        EBTimeLayer = digi_parameters.timeLayerBarrel,
                                        EETimeLayer = digi_parameters.timeLayerEndcap,
                                        correctForVertexZPosition=cms.bool(True),
                                        useMCTruthVertex=cms.bool(True),
                                        recoVertex = cms.InputTag("offlinePrimaryVerticesWithBS"),
                                        simVertex = cms.InputTag("g4SimHits")
                                        )
