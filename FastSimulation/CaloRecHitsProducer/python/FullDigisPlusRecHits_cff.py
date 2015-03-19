import FWCore.ParameterSet.Config as cms
#from FWCore.Modules.printContent_cfi import *

# ECAL Trigger Primitives (needed by SRP)
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *

#from SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_cff import *
# Selective Readout Processor producer

from SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi import *
# Preshower Zero suppression producer
from SimCalorimetry.EcalZeroSuppressionProducers.ecalPreshowerDigis_cfi import *

#from SimCalorimetry.Configuration.ecalDigiSequence_cff import *

#HCAL
from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import * ######fede
hcalSimBlock.hitsProducer = cms.string("famosSimHits")
from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi import *

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import * 
from SimCalorimetry.HcalTrigPrimProducers.hcalTTPDigis_cfi import *

#ECAL reconstruction
from RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi import *
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
#from EventFilter.RawDataCollector.rawDataCollector_cfi import *  
import EventFilter.ESRawToDigi.esRawToDigi_cfi
ecalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()
ecalPreshowerDigis.sourceTag = 'rawDataCollector'

from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *

# HCAL Digi2Raw Raw2Digi
from EventFilter.HcalRawToDigi.HcalDigiToRaw_cfi import * 
#from EventFilter.RawDataCollector.rawDataCollector_cfi import *  
import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi 
hcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()  
hcalDigis.InputLabel = 'rawDataCollector'

##HcalRecHit
from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
hcalOOTPileupESProducer = cms.ESProducer('OOTPileupDBCompatibilityESProducer')
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import *
from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *

#from Validation.HcalDigis.hcalDigisValidationSequence_cff import *
dump = cms.EDAnalyzer("EventContentAnalyzer")

DigiSequence = cms.Sequence((simHcalTriggerPrimitiveDigis * simHcalDigis*simHcalTTPDigis) + (simEcalTriggerPrimitiveDigis*simEcalDigis*simEcalPreshowerDigis ) # Digi
                            *ecalPacker *esDigiToRaw *hcalRawData *rawDataCollector  *ecalPreshowerDigis *ecalDigis *hcalDigis #* printContent
                            )

# Reconstruction
ecalRecHitSequence = cms.Sequence(ecalMultiFitUncalibRecHit*ecalDetIdToBeRecovered*ecalRecHit*ecalPreshowerRecHit)	   	
hcalRecHitSequence = cms.Sequence((hbheprereco+hfreco+horeco)*hbhereco)
hcalRecHitSequencePreTrk = cms.Sequence((hbheprereco+hfreco+horeco))

TotalDigisPlusRecHitsSequence = cms.Sequence(DigiSequence*ecalRecHitSequence*hcalRecHitSequence)
TotalDigisPlusRecHitsSequencePreTrk = cms.Sequence(DigiSequence*ecalRecHitSequence*hcalRecHitSequencePreTrk)

