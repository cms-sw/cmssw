import FWCore.ParameterSet.Config as cms

# This object is used to make changes for different running scenarios. In
# this case for Run 2
from Configuration.StandardSequences.Eras import eras

from EventFilter.CSCTFRawToDigi.csctfpacker_cfi import *
from EventFilter.DTTFRawToDigi.dttfpacker_cfi import *
from EventFilter.GctRawToDigi.gctDigiToRaw_cfi import *
from EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi import *
from EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmPack_cfi import *
from EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi import *
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
import EventFilter.EcalDigiToRaw.ecalDigiToRaw_cfi
ecalPacker = EventFilter.EcalDigiToRaw.ecalDigiToRaw_cfi.ecaldigitorawzerosup.clone()
from EventFilter.ESDigiToRaw.esDigiToRaw_cfi import *
from EventFilter.HcalRawToDigi.HcalDigiToRaw_cfi import *
from EventFilter.CSCRawToDigi.cscPacker_cfi import *
from EventFilter.DTRawToDigi.dtPacker_cfi import *
from EventFilter.RPCRawToDigi.rpcPacker_cfi import *
from EventFilter.CastorRawToDigi.CastorDigiToRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
from EventFilter.L1TRawToDigi.caloStage1Raw_cfi import *
from EventFilter.L1TRawToDigi.caloStage2Raw_cfi import *
#DigiToRaw = cms.Sequence(csctfpacker*dttfpacker*gctDigiToRaw*l1GtPack*l1GtEvmPack*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*rawDataCollector)
DigiToRaw = cms.Sequence(csctfpacker*dttfpacker*gctDigiToRaw*l1GtPack*l1GtEvmPack*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*castorRawData*rawDataCollector)
csctfpacker.lctProducer = "simCscTriggerPrimitiveDigis:MPCSORTED"
csctfpacker.trackProducer = 'simCsctfTrackDigis'
dttfpacker.DTDigi_Source = 'simDtTriggerPrimitiveDigis'
dttfpacker.DTTracks_Source = "simDttfDigis:DTTF"
gctDigiToRaw.rctInputLabel = 'simRctDigis'
gctDigiToRaw.gctInputLabel = 'simGctDigis'
l1GtPack.DaqGtInputTag = 'simGtDigis'
l1GtPack.MuGmtInputTag = 'simGmtDigis'
l1GtEvmPack.EvmGtInputTag = 'simGtDigis'
ecalPacker.Label = 'simEcalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = "simEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "simEcalDigis:eeSrFlags"

##
## Make changes for Run 2
##
def _modifyDigiToRawForStage1L1Trigger( DigiToRaw_object ) :
    DigiToRaw.remove( l1GtEvmPack )
    L1TStage1DigiToRawSeq = cms.Sequence( gctDigiToRaw 
                                          +caloStage1Raw )
    DigiToRaw.replace( gctDigiToRaw, L1TStage1DigiToRawSeq )

eras.stage1L1Trigger.toModify( DigiToRaw, func=_modifyDigiToRawForStage1L1Trigger )
eras.stage1L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("caloStage1Raw")) )

### this appears to be incorrect ! ###
eras.stage1L1Trigger.toModify( gctDigiToRaw, gctInputLabel = 'simCaloStage1LegacyFormatDigis' )

def _modifyDigiToRawForStage2L1Trigger( DigiToRaw_object ) :
    DigiToRaw.remove( l1GtEvmPack )
    DigiToRaw.replace( gctDigiToRaw, caloStage2Raw )
    #DigiToRaw.remove( csctfpacker )
    #DigiToRaw.remove( dttfpacker )
    #DigiToRaw.remove( gctDigiToRaw )
    #DigiToRaw.remove( l1GtPack )

eras.stage2L1Trigger.toModify( DigiToRaw, func=_modifyDigiToRawForStage2L1Trigger )
eras.stage2L1Trigger.toModify( rawDataCollector.RawCollectionList, func = lambda list: list.append(cms.InputTag("caloStage2Raw")) )

# Not in 76X:
#if eras.phase1Pixel.isChosen() :
#    DigiToRaw.remove(siPixelRawData)
#    DigiToRaw.remove(castorRawData)

if eras.fastSim.isChosen() :
    for _entry in [siPixelRawData,SiStripDigiToRaw,castorRawData]:
        DigiToRaw.remove(_entry)
