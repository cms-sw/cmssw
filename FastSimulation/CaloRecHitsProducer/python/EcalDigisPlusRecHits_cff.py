import FWCore.ParameterSet.Config as cms

# ECAL Trigger Primitives (needed by SRP)
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
# Selective Readout Processor producer
from SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi import *

# Preshower Zero suppression producer
from SimCalorimetry.EcalZeroSuppressionProducers.ecalPreshowerDigis_cfi import *

# RCT (Regional Calorimeter Trigger) emulator import 
import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
simRctDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()  
simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'simEcalTriggerPrimitiveDigis' ) ) 
simRctDigis.useHcal = cms.bool(False)

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


from EventFilter.ESDigiToRaw.esDigiToRaw_cfi import *
import EventFilter.ESRawToDigi.esRawToDigi_cfi
ecalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()
ecalPreshowerDigis.sourceTag = 'rawDataCollector'

from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
	  
ecalDigisSequence = cms.Sequence(simEcalTriggerPrimitiveDigis*simEcalDigis*simEcalPreshowerDigis* # Digi
			  simRctDigis*							           # L1Simulation
                          ecalPacker*esDigiToRaw*rawDataCollector* ecalPreshowerDigis*ecalDigis)	

ecalRecHitSequence = cms.Sequence(ecalGlobalUncalibRecHit*ecalDetIdToBeRecovered*ecalRecHit*ecalPreshowerRecHit)	   # Reconstruction	

ecalDigisPlusRecHitSequence = cms.Sequence(ecalDigisSequence*ecalRecHitSequence)	           # Reconstruction	
			  
