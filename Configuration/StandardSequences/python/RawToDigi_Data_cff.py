import FWCore.ParameterSet.Config as cms

#
# David Lange, Bryan Dahmes LLNL
# 30 April, 2007
#
#--- Allow for multiple calls to the database ---#
from CondCore.DBCommon.CondDBSetup_cfi import *
#--- Geometry Setup ---#
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from HLTrigger.Configuration.rawToDigi.EcalGeometrySetup_cff import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
#-------------------------#
#--- RawToDigi modules ---#
#-------------------------#
#--- L1 Setup ---#
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff import *
from L1TriggerConfig.GctConfigProducers.L1GctConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff import *
from L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff import *

import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
#--- CSC TF ---#
csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
#--- DT TF ---#
dttfDigis = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone()
import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
#--- GCT/GMT ---#
gctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone()
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
#--- SiPixel ---#
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
RawToDigi = cms.Sequence(csctfDigis+dttfDigis+gctDigis+gtDigis+siPixelDigis+siStripDigis+ecalDigis+ecalPreshowerDigis+hcalDigis+muonCSCDigis+muonDTDigis+muonRPCDigis)
gtDigis.DaqGtInputTag = 'source'
siPixelDigis.InputLabel = 'source'
siStripDigis.ProductLabel = 'source'
ecalDigis.DoRegional = False
muonCSCDigis.UseExaminer = False

