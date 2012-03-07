import FWCore.ParameterSet.Config as cms

#To include the ECAL RecHit containment corrections (the famous 0.97 factor)
from SimCalorimetry.EcalSimProducers.ecalNotContainmentSim_cff import *

# This includes is needed for the ECAL digis
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
from SimCalorimetry.HcalSimProducers.hcalSimParameters_cfi import *

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
                                                       HighNoiseParameters = cms.vdouble(2.56,0.51,0.0086),
                                                       Threshold = cms.double(0.1),
						       SRThreshold = cms.double(1.),
#						       SREtaSize = cms.untracked.int32(1),
#						       SRPhiSize = cms.untracked.int32(1),
                                                       Refactor = cms.double(1.),
                                                       Refactor_mean = cms.double(1.),
                                                       MixedSimHits = cms.InputTag("mix","famosSimHitsEcalHitsEB"),
                                                       ContFact = cms.PSet(ecal_notCont_sim)),
                                                       
                                                       ECALEndcap = cms.PSet(
                                                       Noise = cms.double(-1.),
                                                       NoiseADC = cms.double(2.32),
                                                       HighNoiseParameters = cms.vdouble(4.77,1.65,2.7,5.1),
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


hbhereco = cms.EDProducer("CaloRecHitsProducer",
                          InputRecHitCollectionTypes = cms.vuint32(4),
                          OutputRecHitCollections = cms.vstring(""),
                          doDigis = cms.bool(False),
                          doMiscalib = cms.bool(False),
                          
                          RecHitsFactory = cms.PSet(
                                           HCAL = cms.PSet(
                                           Noise = cms.vdouble(-1.,-1.),
                                           NoiseCorrectionFactor = cms.vdouble(1.39,1.32),
                                           Threshold = cms.vdouble(-1.,-1.),
                                           MixedSimHits = cms.InputTag("mix","famosSimHitsHcalHits"),
                                           EnableSaturation = cms.bool(True),
                                           Refactor = cms.double(1.),
                                           Refactor_mean = cms.double(1.),
                                           fileNameHcal = cms.string('hcalmiscalib_0.0.xml'))))


horeco = cms.EDProducer("CaloRecHitsProducer",
                          InputRecHitCollectionTypes = cms.vuint32(5),
                          OutputRecHitCollections = cms.vstring(""),
                          doDigis = cms.bool(False),
                          doMiscalib = cms.bool(False),
                          
                          RecHitsFactory = cms.PSet(
                                           HCAL = cms.PSet(
                                           Noise = cms.vdouble(-1.),
                                           NoiseCorrectionFactor = cms.vdouble(3.),#1.17 to be tuned
                                           Threshold = cms.vdouble(-1.5),
                                           MixedSimHits = cms.InputTag("mix","famosSimHitsHcalHits"),
                                           EnableSaturation = cms.bool(True),
                                           Refactor = cms.double(1.),
                                           Refactor_mean = cms.double(1.),
                                           fileNameHcal = cms.string('hcalmiscalib_0.0.xml'))))

hfreco = cms.EDProducer("CaloRecHitsProducer",
                        InputRecHitCollectionTypes = cms.vuint32(6),
                        OutputRecHitCollections = cms.vstring(""),
                        doDigis = cms.bool(False),
                        doMiscalib = cms.bool(False),
                        
                        RecHitsFactory = cms.PSet(
                                           HCAL = cms.PSet(
                                           Noise = cms.vdouble(-1.),
                                           NoiseCorrectionFactor = cms.vdouble(2.51),
                                           Threshold = cms.vdouble(-0.5),
                                           MixedSimHits = cms.InputTag("mix","famosSimHitsHcalHits"),
                                           EnableSaturation = cms.bool(True),
                                           Refactor = cms.double(1.),
                                           Refactor_mean = cms.double(1.),
                                           fileNameHcal = cms.string('hcalmiscalib_0.0.xml'))))


simHcalTriggerPrimitiveDigis.peakFilter = False
simHcalTriggerPrimitiveDigis.inputLabel =  cms.VInputTag(cms.InputTag('hbhereco'), cms.InputTag('hfreco'))
'hbhereco'
simEcalTriggerPrimitiveDigis.Famos = True
simEcalTriggerPrimitiveDigis.Label = 'ecalRecHit'

caloRecHits = cms.Sequence(ecalRecHit*ecalPreshowerRecHit*hbhereco*horeco*hfreco)
