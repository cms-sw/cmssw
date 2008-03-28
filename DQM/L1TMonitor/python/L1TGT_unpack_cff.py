import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff import *
from L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff import *
from EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi import *
# input tag
# packed GT DAQ record starting from DAQ
l1GtUnpack.DaqGtInputTag = 'source'

