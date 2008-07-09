import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff import *
from L1TriggerConfig.L1ScalesProducers.L1MuGMTScalesConfig_cff import *
from L1TriggerConfig.GMTConfigProducers.L1MuGMTParametersConfig_cff import *
from EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi import *
# L1 GT Packer
#replace l1GtUnpack.DaqGtInputTag = l1GtPack
# packed GT DAQ record starting from dumped Spy data
# replace l1GtUnpack.DaqGtInputTag = vmeToRaw
# Active Boards
# GTFE only in the record
#    replace l1GtUnpack.ActiveBoardsMask = 0x0000
# GTFE + FDL
#replace l1GtUnpack.ActiveBoardsMask = 0x0001
# GTFE + GMT
#replace l1GtUnpack.ActiveBoardsMask = 0x0100
# GTFE + FDL + GMT
#replace l1GtUnpack.ActiveBoardsMask = 0x0101
# BxInEvent to be unpacked
# all available BxInEvent
#replace l1GtUnpack.UnpackBxInEvent = -1
# BxInEvent = 0 (L1A)
#    replace l1GtUnpack.UnpackBxInEvent = 1
# 3 BxInEvent (F, 0, 1)
#replace l1GtUnpack.UnpackBxInEvent = 3
from EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi import *
# input tag
# packed GT DAQ record starting from DAQ
l1GtUnpack.DaqGtInputTag = 'source'
l1GtEvmUnpack.EvmGtInputTag = 'source'

