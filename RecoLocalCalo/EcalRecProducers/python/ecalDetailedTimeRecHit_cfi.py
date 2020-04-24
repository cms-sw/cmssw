import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalTimeDigiParameters_cff import *

ecalDetailedTimeRecHit = cms.EDProducer("EcalDetailedTimeRecHitProducer",
                                        EERecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                        EBRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                        EETimeDigiCollection = cms.InputTag("mix","EETimeDigi"),
                                        EBTimeDigiCollection = cms.InputTag("mix","EBTimeDigi"),
                                        EBDetailedTimeRecHitCollection = cms.string('EcalRecHitsEB'),
                                        EEDetailedTimeRecHitCollection = cms.string('EcalRecHitsEE'),
                                        EBTimeLayer = ecal_time_digi_parameters.timeLayerBarrel,
                                        EETimeLayer = ecal_time_digi_parameters.timeLayerEndcap,
                                        correctForVertexZPosition=cms.bool(False),
                                        useMCTruthVertex=cms.bool(False),
                                        recoVertex = cms.InputTag("offlinePrimaryVerticesWithBS"),
                                        simVertex = cms.InputTag("g4SimHits")
                                        )
