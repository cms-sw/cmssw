import FWCore.ParameterSet.Config as cms

# Setup
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff import *
from L1TriggerConfig.GctConfigProducers.L1GctConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff import *
from L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff import *
import EventFilter.CSCTFRawToDigi.csctfunpacker_cfi
# CSC TF
csctfDigis = EventFilter.CSCTFRawToDigi.csctfunpacker_cfi.csctfunpacker.clone()
import EventFilter.DTTFRawToDigi.dttfunpacker_cfi
# DT TF
dttfDigis = EventFilter.DTTFRawToDigi.dttfunpacker_cfi.dttfunpacker.clone()
import EventFilter.GctRawToDigi.l1GctHwDigis_cfi
# GCT
gctDigis = EventFilter.GctRawToDigi.l1GctHwDigis_cfi.l1GctHwDigis.clone()
import EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi
# GT
gtDigis = EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi.l1GtUnpack.clone()
from EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi import *
gtRecords = cms.Sequence(gtDigis*l1GtRecord)
L1RawToDigi = cms.Sequence(csctfDigis+dttfDigis+gctDigis+gtRecords)
csctfDigis.producer = 'rawDataCollector'
dttfDigis.DTTF_FED_Source = 'rawDataCollector'
gctDigis.inputLabel = 'rawDataCollector'
gtDigis.DaqGtInputTag = 'rawDataCollector'

