import FWCore.ParameterSet.Config as cms
from FWCore.Modules.printContent_cfi import *

# ECAL Trigger Primitives (needed by SRP)
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
# Selective Readout Processor producer
from SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi import *

#HCAL
from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi import *
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import * 
from SimCalorimetry.HcalTrigPrimProducers.hcalTTPDigis_cfi import *

# RCT (Regional Calorimeter Trigger) emulator import for both 
import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
simRctDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()  
simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'simEcalTriggerPrimitiveDigis' ) )
simRctDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'simHcalTriggerPrimitiveDigis' ) ) 
simRctDigis.useEcal = cms.bool(True)
simRctDigis.useHcal = cms.bool(True)

#ECAL reconstruction
from RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi import *

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
# HCAL Digi2Raw Raw2Digi
from EventFilter.HcalRawToDigi.HcalDigiToRaw_cfi import * 
from EventFilter.RawDataCollector.rawDataCollector_cfi import *  
import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi 
hcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()  
hcalDigis.InputLabel = 'rawDataCollector'

##HcalRecHit
from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import *
from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *

DigiSequence = cms.Sequence(((simHcalTriggerPrimitiveDigis*simHcalDigis*simHcalTTPDigis) + (simEcalTriggerPrimitiveDigis*simEcalDigis )) # Digi
                            *simRctDigis*                            						           # L1Simulation
                            (ecalPacker+ hcalRawData)* rawDataCollector * (ecalDigis + hcalDigis) #* printContent
                            )


# Reconstruction
ecalRecHitSequence = cms.Sequence(ecalGlobalUncalibRecHit*ecalDetIdToBeRecovered*ecalRecHit)	   	
hcalRecHitSequence = cms.Sequence((hbheprereco+hfreco+horeco)*hbhereco)

TotalDigisPlusRecHitsSequence = cms.Sequence(DigiSequence*ecalRecHitSequence*hcalRecHitSequence)

