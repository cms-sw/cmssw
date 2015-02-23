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
#DigiToRaw = cms.Sequence(csctfpacker*dttfpacker*gctDigiToRaw*l1GtPack*l1GtEvmPack*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*rawDataCollector)
DigiToRaw = cms.Sequence(csctfpacker*dttfpacker*gctDigiToRaw*l1GtPack*l1GtEvmPack*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*castorRawData*rawDataCollector)
csctfpacker.lctProducer = "simCscTriggerPrimitiveDigis:MPCSORTED"
csctfpacker.trackProducer = 'simCsctfTrackDigis'
dttfpacker.DTDigi_Source = 'simDtTriggerPrimitiveDigis'
dttfpacker.DTTracks_Source = "simDttfDigis:DTTF"
gctDigiToRaw.rctInputLabel = 'simRctDigis'
gctDigiToRaw.gctInputLabel = 'simGctDigis'
# Change only if using the Stage 1 trigger
eras.stage1L1Trigger.toModify( gctDigiToRaw, gctInputLabel = 'simCaloStage1LegacyFormatDigis' )

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
def _modifyDigiToRawForStage1L1Trigger( theProcess ) :
    """
    Modifies the DigiToRaw sequence for running in Run 2
    """
    theProcess.load("L1Trigger.L1TCommon.l1tDigiToRaw_cfi")
    # Note that this function is applied before the objects in this file are added
    # to the process. So things declared in this file should be used "bare", i.e.
    # not with "theProcess." in front of them. l1tDigiToRaw is an exception because
    # it is not declared in this file but loaded into the process in the "load"
    # statement above.
    l1tDigiToRawSeq = cms.Sequence( gctDigiToRaw + theProcess.l1tDigiToRaw )
    DigiToRaw.replace( gctDigiToRaw, l1tDigiToRawSeq )

# A unique name is required for this object, so I'll call it "modify<python filename>ForRun2_"
modifyConfigurationStandardSequencesDigiToRawForRun2_ = eras.stage1L1Trigger.makeProcessModifier( _modifyDigiToRawForStage1L1Trigger )
