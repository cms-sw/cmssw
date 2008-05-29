import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
#from HLTrigger.Configuration.rawToDigi.EcalGeometrySetup_cff import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff import *
from L1TriggerConfig.GctConfigProducers.L1GctConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff import *
from L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff import *
import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
dttfDigis = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone()
import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
gctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone()
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
from EventFilter.SiStripRawToDigi.SiStripRawToDigis_standard_cff import *
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
import EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi
ecalDigis = EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi.ecalEBunpacker.clone()
import EventFilter.ESRawToDigi.esRawToDigi_cfi
ecalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()
import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
hcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()
import EventFilter.CSCRawToDigi.cscUnpacker_cfi
muonCSCDigis = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone()
import EventFilter.DTRawToDigi.dtunpacker_cfi
muonDTDigis = EventFilter.DTRawToDigi.dtunpacker_cfi.muonDTDigis.clone()
import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
muonRPCDigis = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone()
RawToDigi = cms.Sequence(csctfDigis+dttfDigis+gctDigis+gtDigis+siPixelDigis+SiStripRawToDigis+ecalDigis+ecalPreshowerDigis+hcalDigis+muonCSCDigis+muonDTDigis+muonRPCDigis)
csctfDigis.producer = 'rawDataCollector'
dttfDigis.DTTF_FED_Source = 'rawDataCollector'
gctDigis.inputLabel = 'rawDataCollector'
gtDigis.DaqGtInputTag = 'rawDataCollector'
siPixelDigis.InputLabel = 'rawDataCollector'
ecalDigis.DoRegional = False
ecalDigis.InputLabel = 'rawDataCollector'
ecalPreshowerDigis.Label = 'rawDataCollector'
hcalDigis.InputLabel = 'rawDataCollector'
muonCSCDigis.InputObjects = 'rawDataCollector'
muonCSCDigis.UseExaminer = False
muonDTDigis.fedColl = 'rawDataCollector'
muonRPCDigis.InputLabel = 'rawDataCollector'

