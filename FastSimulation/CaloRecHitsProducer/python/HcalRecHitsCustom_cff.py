import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
from SimCalorimetry.HcalSimProducers.hcalSimParameters_cfi import *

simHcalTriggerPrimitiveDigis.peakFilter = False
simHcalTriggerPrimitiveDigis.inputLabel =  cms.VInputTag(cms.InputTag('hbhereco'), cms.InputTag('hfreco'))
'hbhereco'

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



