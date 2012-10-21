import FWCore.ParameterSet.Config as cms

####Hcal Digis
#from FWCore.Modules.printContent_cfi import *

## import SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi 
## hcalSimBlockFastSim = SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi.hcalSimBlock.clone()
## hcalSimBlockFastSim.hitsProducer = cms.string('famosSimHits')
## simHcalUnsuppressedDigis = cms.EDProducer("HcalDigiProducer",
##                                            hcalSimBlockFastSim)
from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi import *
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import * 
from SimCalorimetry.HcalTrigPrimProducers.hcalTTPDigis_cfi import *

# RCT (Regional Calorimeter Trigger) emulator import 
import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
simRctDigisHCAL = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()  
simRctDigisHCAL.hcalDigis = cms.VInputTag( cms.InputTag( 'simHcalTriggerPrimitiveDigis' ) ) 
simRctDigisHCAL.useEcal = cms.bool(False)

#Digi2Raw Raw2Digi
from EventFilter.HcalRawToDigi.HcalDigiToRaw_cfi import * 
from EventFilter.RawDataCollector.rawDataCollector_cfi import *  
import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi 
hcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()  
hcalDigis.InputLabel = 'rawDataCollector'

hcalDigisSequence = cms.Sequence(simHcalTriggerPrimitiveDigis*simHcalDigis*simHcalTTPDigis
                                *simRctDigisHCAL
                                *hcalRawData*rawDataCollector*hcalDigis)

##HcalRecHit
from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import *
from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *

hcalRecHitSequence = cms.Sequence((hbheprereco+hfreco+horeco)*hbhereco)

hcalDigisPlusRecHitsSequence = cms.Sequence(hcalDigisSequence*hcalRecHitSequence)

