import FWCore.ParameterSet.Config as cms

#To include the ECAL RecHit containment corrections (the famous 0.97 factor)
from SimCalorimetry.EcalSimProducers.ecalNotContainmentSim_cff import *

# This includes is needed for the ECAL digis
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *

ecalRecHit = cms.EDProducer("CaloRecHitsProducer",
                            InputRecHitCollectionTypes = cms.vuint32(2, 3),
                            OutputRecHitCollections = cms.vstring('EcalRecHitsEB', 
                                                                  'EcalRecHitsEE'),
                            doDigis = cms.bool(False),
                            doMiscalib = cms.bool(False),

                            RecHitsFactory = cms.PSet(
                                                       ECALBarrel = cms.PSet(
                                                       Noise = cms.double(-1.),
                                                       NoiseADC = cms.double(1.054),
                                                       HighNoiseParameters = cms.vdouble(0.04,0.51,0.016),
                                                       Threshold = cms.double(0.1),
						       SRThreshold = cms.double(1.),
                                                       Refactor = cms.double(1.),
                                                       Refactor_mean = cms.double(1.),
                                                       MixedSimHits = cms.InputTag("mix","famosSimHitsEcalHitsEB"),
                                                       ContFact = cms.PSet(ecal_notCont_sim)),
                                                       
                                                       ECALEndcap = cms.PSet(
                                                       Noise = cms.double(-1.),
                                                       NoiseADC = cms.double(2.32),
                                                       HighNoiseParameters = cms.vdouble(5.72,1.65,2.7,6.1),
                                                       Threshold = cms.double(.32), 
                                                       SRThreshold = cms.double(1.),
                                                       Refactor = cms.double(1.),
                                                       Refactor_mean = cms.double(1.),
                                                       MixedSimHits = cms.InputTag("mix","famosSimHitsEcalHitsEE"),
                                                       ContFact = cms.PSet(ecal_notCont_sim)),
                                                       ))


ecalPreshowerRecHit =  cms.EDProducer("CaloRecHitsProducer",
                                      InputRecHitCollectionTypes = cms.vuint32(1),
                                      OutputRecHitCollections = cms.vstring('EcalRecHitsES'),
                                      doDigis = cms.bool(False),
                                      doMiscalib = cms.bool(False),

                                      RecHitsFactory = cms.PSet(
                                                       ECALPreshower = cms.PSet(
                                                       Noise = cms.double(1.5e-05),
                                                       Threshold = cms.double(4.5e-05),
                                                       MixedSimHits = cms.InputTag("mix","famosSimHitsEcalHitsES"))))


simEcalTriggerPrimitiveDigis.Famos = True
simEcalTriggerPrimitiveDigis.Label = 'ecalRecHit'
