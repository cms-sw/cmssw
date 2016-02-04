import FWCore.ParameterSet.Config as cms

# unpack L1 GT DAQ record from data taking
from EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi import *
l1GtUnpack.DaqGtInputTag = 'source'


# unpack L1 GT EVM record from data taking
from EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi import *
l1GtEvmUnpack.EvmGtInputTag = 'source'

