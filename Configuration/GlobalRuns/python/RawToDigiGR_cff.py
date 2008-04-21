# The following comments couldn't be translated into the new config version:

#                        l1GctEmulDigis &

#                        siPixelDigis &
#                        SiStripDigis &

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
l1CscTfEmulDigis = copy.deepcopy(csctfunpacker)
import copy
from EventFilter.DTTFRawToDigi.dttfunpacker_cfi import *
#--- DT TF ---#
l1DttfEmulDigis = copy.deepcopy(dttfunpacker)
import copy
from EventFilter.GctRawToDigi.l1GctHwDigis_cfi import *
#--- GCT/GMT ---#
l1GctEmulDigis = copy.deepcopy(l1GctHwDigis)
import copy
from EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi import *
l1GmtEmulDigis = copy.deepcopy(l1GtUnpack)
#--- SiPixel ---#
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
#--- SiStrip ---#
#workaround for still wrong label names 
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
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
#--- DT ---#
#module dtunpacker = muonDTDigis from "EventFilter/DTRawToDigi/data/dtunpacker.cfi"
from EventFilter.DTRawToDigi.dtunpacker_cfi import *
import copy
from EventFilter.RPCRawToDigi.rpcUnpacker_cfi import *
#--- RPC ---#
muonRPCDigis = copy.deepcopy(rpcunpacker)
RawToDigi = cms.Sequence(l1CscTfEmulDigis+l1DttfEmulDigis+l1GmtEmulDigis+ecalDigis+ecalPreshowerDigis+hcalDigis+muonCSCDigis+muonDTDigis+muonRPCDigis)
l1GmtEmulDigis.DaqGtInputTag = 'source'
siPixelDigis.InputLabel = 'source'
SiStripDigis.ProductLabel = 'source'
ecalDigis.DoRegional = False
muonCSCDigis.UseExaminer = False

