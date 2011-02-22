import FWCore.ParameterSet.Config as cms

#
# David Lange, Bryan Dahmes LLNL
# 30 April, 2007
#
#--- Allow for multiple calls to the database ---#
from CondCore.DBCommon.CondDBSetup_cfi import *
#--- Geometry Setup ---#
#from HLTrigger.Configuration.rawToDigi.EcalGeometrySetup_cff import *
#-------------------------#
#--- RawToDigi modules ---#
#-------------------------#

import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
#--- CSC TF ---#
import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
dttfDigis = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone()
#--- DT TF ---#
import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
gctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone()
#--- GCT/GMT ---#
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
import EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi
gtEvmDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi.l1GtEvmUnpack.clone()#--- SiPixel ---#
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
#--- SiStrip ---#
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
#--- Ecal ---#
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
import EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi
ecalDigis = EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi.ecalEBunpacker.clone()
import EventFilter.ESRawToDigi.esRawToDigi_cfi
#--- Ecal Preshower ---#
ecalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()
import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
#--- Hcal ---#
hcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()
import EventFilter.CSCRawToDigi.cscUnpacker_cfi
#--- CSC ---#
muonCSCDigis = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone()
#--- DT ---#
#module dtunpacker = muonDTDigis from "EventFilter/DTRawToDigi/data/dtunpacker.cfi"
from EventFilter.DTRawToDigi.dtunpacker_cfi import *
import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
#--- RPC ---#
muonRPCDigis = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone()
#--- castor ---#
from  EventFilter.CastorRawToDigi.CastorRawToDigi_cff import *
castorDigis = EventFilter.CastorRawToDigi.CastorRawToDigi_cfi.castorDigis.clone( FEDs = cms.untracked.vint32(690,691,692) )
#--- Scalers ---#
from EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi import *


RawToDigi = cms.Sequence(csctfDigis+dttfDigis+gctDigis+gtDigis+gtEvmDigis+siPixelDigis+siStripDigis+ecalDigis+ecalPreshowerDigis+hcalDigis+muonCSCDigis+muonDTDigis+muonRPCDigis+castorDigis+scalersRawToDigi)

RawToDigi_woGCT = cms.Sequence(csctfDigis+dttfDigis+gtDigis+gtEvmDigis+siPixelDigis+siStripDigis+ecalDigis+ecalPreshowerDigis+hcalDigis+muonCSCDigis+muonDTDigis+muonRPCDigis+castorDigis+scalersRawToDigi)

gtDigis.DaqGtInputTag = 'source'
gtEvmDigis.EvmGtInputTag = 'source'
siPixelDigis.InputLabel = 'source'
siStripDigis.ProductLabel = 'source'
castorDigis.InputLabel = 'source'
ecalDigis.DoRegional = False
