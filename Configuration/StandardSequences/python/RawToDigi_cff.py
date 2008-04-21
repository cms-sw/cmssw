import FWCore.ParameterSet.Config as cms

#
# David Lange, Bryan Dahmes LLNL
# 30 April, 2007
#
#--- Allow for multiple calls to the database ---#
from CondCore.DBCommon.CondDBSetup_cfi import *
#--- Geometry Setup ---#
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
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
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff import *
from L1TriggerConfig.GctConfigProducers.L1GctConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff import *
from L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff import *
import copy
from EventFilter.CSCTFRawToDigi.csctfunpacker_cfi import *
#--- CSC TF ---#
csctfDigis = copy.deepcopy(csctfunpacker)
import copy
from EventFilter.DTTFRawToDigi.dttfunpacker_cfi import *
#--- DT TF ---#
dttfDigis = copy.deepcopy(dttfunpacker)
import copy
from EventFilter.GctRawToDigi.l1GctHwDigis_cfi import *
#--- GCT ---#
gctDigis = copy.deepcopy(l1GctHwDigis)
import copy
from EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi import *
#--- GT ---#
gtDigis = copy.deepcopy(l1GtUnpack)
#--- SiPixel ---#
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
#--- SiStrip ---#
from EventFilter.SiStripRawToDigi.SiStripRawToDigis_standard_cff import *
#--- Ecal ---#
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
import copy
from EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi import *
ecalDigis = copy.deepcopy(ecalEBunpacker)
import copy
from EventFilter.ESRawToDigi.esRawToDigi_cfi import *
#--- Ecal Preshower ---#
ecalPreshowerDigis = copy.deepcopy(esRawToDigi)
import copy
from EventFilter.HcalRawToDigi.HcalRawToDigi_cfi import *
#--- Hcal ---#
hcalDigis = copy.deepcopy(hcalDigis)
import copy
from EventFilter.CSCRawToDigi.cscUnpacker_cfi import *
#--- CSC ---#
muonCSCDigis = copy.deepcopy(muonCSCDigis)
import copy
from EventFilter.DTRawToDigi.dtunpacker_cfi import *
#--- DT ---#
muonDTDigis = copy.deepcopy(muonDTDigis)
import copy
from EventFilter.RPCRawToDigi.rpcUnpacker_cfi import *
#--- RPC ---#
muonRPCDigis = copy.deepcopy(rpcunpacker)
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

