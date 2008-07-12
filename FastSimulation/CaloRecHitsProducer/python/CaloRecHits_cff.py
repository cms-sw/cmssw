import FWCore.ParameterSet.Config as cms

#To include the ECAL RecHit containment corrections (the famous 0.97 factor)
from SimCalorimetry.EcalSimProducers.ecalNotContainmentSim_cff import *

# Thes includes are needed for the HCAL digi
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
from SimCalorimetry.HcalSimProducers.hcalSimParameters_cfi import *

# This includes is needed for the ECAL digis
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
caloRecHits = cms.EDProducer("CaloRecHitsProducer",
    RecHitsFactory = cms.PSet(
        ECALPreshower = cms.PSet(
            Threshold = cms.double(4.5e-05),
            MixedSimHits = cms.InputTag("mix","famosSimHitsEcalHitsES"),
            Noise = cms.double(1.5e-05)
        ),
        ECALEndcap = cms.PSet(
            Threshold = cms.double(-999.0),
            MixedSimHits = cms.InputTag("mix","famosSimHitsEcalHitsEE"),
            Refactor_mean = cms.double(1.0),
            Noise = cms.double(0.15),
            Refactor = cms.double(1.0)
        ),
        EErechitCollection = cms.string('EcalRecHitsEE'),
        ESrechitCollection = cms.string('EcalRecHitsES'),
        EBrechitCollection = cms.string('EcalRecHitsEB'),
        doMiscalib = cms.bool(False), ## does not apply in the PS

        HCAL = cms.PSet(
            NoiseHO = cms.double(0.17),
            SaturationHB = cms.double(14000.0),
            MixedSimHits = cms.InputTag("mix","famosSimHitsHcalHits"),
            NoiseHB = cms.double(0.23),
            NoiseHE = cms.double(0.31),
            Refactor = cms.double(1.0),
            NoiseHF = cms.double(0.0),
            SaturationHO = cms.double(14000.0),
            ThresholdHO = cms.double(1.1),
            fileNameHcal = cms.string('hcalmiscalib_startup.xml'),
            ThresholdHB = cms.double(0.9),
            SaturationHE = cms.double(14000.0),
            SaturationHF = cms.double(14000.0),
            ThresholdHF = cms.double(0.5),
            Refactor_mean = cms.double(1.0),
            ThresholdHE = cms.double(1.4)
        ),
        ECALBarrel = cms.PSet(
            Threshold = cms.double(-999.0),
            MixedSimHits = cms.InputTag("mix","famosSimHitsEcalHitsEB"),
            Refactor_mean = cms.double(1.0),
            Noise = cms.double(0.04),
            Refactor = cms.double(1.0)
        ),
        doDigis = cms.bool(False) ## not for PS

    ),
    #the famous 0.97 factor..
    ContFact = cms.PSet(
        ecal_notCont_sim
    ),
    hcalSimParam = cms.PSet(
        hcalSimParameters
    )
)

simHcalTriggerPrimitiveDigis.inputLabel = 'caloRecHits'
simHcalTriggerPrimitiveDigis.peakFilter = False
simEcalTriggerPrimitiveDigis.Famos = True
simEcalTriggerPrimitiveDigis.Label = 'caloRecHits'

