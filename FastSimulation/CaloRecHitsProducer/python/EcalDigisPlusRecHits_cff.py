import FWCore.ParameterSet.Config as cms

#To include the ECAL RecHit containment corrections (the famous 0.97 factor)
from SimCalorimetry.EcalSimProducers.ecalNotContainmentSim_cff import *

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
from SimCalorimetry.HcalSimProducers.hcalSimParameters_cfi import *

# This includes is needed for the ECAL digis
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *

# unsuppressed digis simulation - fast preshower
from SimCalorimetry.EcalSimProducers.ecaldigi_cfi import *

# ECAL Trigger Primitives (needed by SRP)
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
# Selective Readout Processor producer
#import SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi 
#ecalDigis = SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi.simEcalDigis.clone()
from SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi import *

# RCT (Regional Calorimeter Trigger) emulator import 
import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
simRctDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()  
simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'simEcalTriggerPrimitiveDigis' ) ) 

#ECAL reconstruction
from RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi import *
simEcalUnsuppressedDigis.hitsProducer = cms.string('famosSimHits')
ecal_digi_parameters.hitsProducer = cms.string('famosSimHits')

import EventFilter.EcalDigiToRaw.ecalDigiToRaw_cfi
ecalPacker = EventFilter.EcalDigiToRaw.ecalDigiToRaw_cfi.ecaldigitorawzerosup.clone()
ecalPacker.Label = 'simEcalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = "simEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "simEcalDigis:eeSrFlags"

from EventFilter.RawDataCollector.rawDataCollector_cfi import *

import EventFilter.EcalRawToDigi.EcalUnpackerData_cfi
ecalDigis = EventFilter.EcalRawToDigi.EcalUnpackerData_cfi.ecalEBunpacker.clone()
ecalDigis.InputLabel = 'rawDataCollector'
#ecalDigis.InputLabel = 'source'

##### THIS IS JUST TEMPORARY, THE ECAL PRESHOWER MUST EVENTUALLY BE MOVED TO REAL DIGITIZER TOO
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
#####


ecalDigisPlusRecHitSequence = cms.Sequence(simEcalUnsuppressedDigis*simEcalTriggerPrimitiveDigis*simEcalDigis* # Digi
			  simRctDigis*							           # L1Simulation
                          ecalPacker*rawDataCollector*ecalDigis*                                   #Digi2raw raw2digi
			  ecalGlobalUncalibRecHit*ecalDetIdToBeRecovered*ecalRecHit)	           # Reconstruction	
			  
