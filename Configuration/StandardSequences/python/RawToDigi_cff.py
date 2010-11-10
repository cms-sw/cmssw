import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
#from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
#from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
#from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
#from HLTrigger.Configuration.rawToDigi.EcalGeometrySetup_cff import *
#from Geometry.DTGeometry.dtGeometry_cfi import *
#from Geometry.RPCGeometry.rpcGeometry_cfi import *

import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
dttfDigis = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone()
import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
gctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone()
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
import EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi
gtEvmDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi.l1GtEvmUnpack.clone()
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
from EventFilter.CastorRawToDigi.CastorRawToDigi_cff import *
castorDigis = EventFilter.CastorRawToDigi.CastorRawToDigi_cfi.castorDigis.clone( FEDs = cms.untracked.vint32(690,691,692) )

from EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi import *

RawToDigi = cms.Sequence(csctfDigis+dttfDigis+gctDigis+gtDigis+gtEvmDigis+siPixelDigis+SiStripRawToDigis+ecalDigis+ecalPreshowerDigis+hcalDigis+muonCSCDigis+muonDTDigis+muonRPCDigis+castorDigis+scalersRawToDigi)

scalersRawToDigi.scalersInputTag = 'rawDataCollector'
csctfDigis.producer = 'rawDataCollector'
dttfDigis.DTTF_FED_Source = 'rawDataCollector'
gctDigis.inputLabel = 'rawDataCollector'
gtDigis.DaqGtInputTag = 'rawDataCollector'
siPixelDigis.InputLabel = 'rawDataCollector'
ecalDigis.DoRegional = False
ecalDigis.InputLabel = 'rawDataCollector'
ecalPreshowerDigis.sourceTag = 'rawDataCollector'
hcalDigis.InputLabel = 'rawDataCollector'
muonCSCDigis.InputObjects = 'rawDataCollector'
muonDTDigis.inputLabel = 'rawDataCollector'
muonRPCDigis.InputLabel = 'rawDataCollector'
gtEvmDigis.EvmGtInputTag = 'rawDataCollector'
castorDigis.InputLabel = 'rawDataCollector'
