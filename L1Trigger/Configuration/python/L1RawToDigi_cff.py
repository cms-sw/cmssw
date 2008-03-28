import FWCore.ParameterSet.Config as cms

# Setup
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff import *
from L1TriggerConfig.GctConfigProducers.L1GctConfig_cff import *
from L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff import *
from L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff import *
import copy
from EventFilter.CSCTFRawToDigi.csctfunpacker_cfi import *
# CSC TF
csctfDigis = copy.deepcopy(csctfunpacker)
import copy
from EventFilter.DTTFRawToDigi.dttfunpacker_cfi import *
# DT TF
dttfDigis = copy.deepcopy(dttfunpacker)
import copy
from EventFilter.GctRawToDigi.l1GctHwDigis_cfi import *
# GCT
gctDigis = copy.deepcopy(l1GctHwDigis)
import copy
from EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi import *
# GT
gtDigis = copy.deepcopy(l1GtUnpack)
from EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi import *
gtRecords = cms.Sequence(gtDigis*l1GtRecord)
L1RawToDigi = cms.Sequence(csctfDigis+dttfDigis+gctDigis+gtRecords)
csctfDigis.producer = 'rawDataCollector'
dttfDigis.DTTF_FED_Source = 'rawDataCollector'
gctDigis.inputLabel = 'rawDataCollector'
gtDigis.DaqGtInputTag = 'rawDataCollector'

