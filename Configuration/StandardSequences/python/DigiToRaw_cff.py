import FWCore.ParameterSet.Config as cms

from EventFilter.CSCTFRawToDigi.csctfpacker_cfi import *
from EventFilter.DTTFRawToDigi.dttfpacker_cfi import *
from EventFilter.GctRawToDigi.gctDigiToRaw_cfi import *
from EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi import *
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
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
DigiToRaw = cms.Sequence(csctfpacker*dttfpacker*gctDigiToRaw*l1GtPack*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*rawDataCollector)
csctfpacker.lctProducer = cms.InputTag("simCscTriggerPrimitiveDigis","MPCSORTED")
csctfpacker.trackProducer = 'simCsctfTrackDigis'
dttfpacker.DTDigi_Source = 'simDtTriggerPrimitiveDigis'
dttfpacker.DTTracks_Source = cms.InputTag("simDttfDigis","DTTF")
gctDigiToRaw.rctInputLabel = 'simRctDigis'
gctDigiToRaw.gctInputLabel = 'simGctDigis'
l1GtPack.DaqGtInputTag = 'simGtDigis'
l1GtPack.MuGmtInputTag = 'simGmtDigis'
ecalPacker.Label = 'simEcalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = cms.InputTag("simEcalDigis","ebSrFlags")
ecalPacker.labelEESRFlags = cms.InputTag("simEcalDigis","eeSrFlags")

