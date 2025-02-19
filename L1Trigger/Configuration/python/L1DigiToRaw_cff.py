import FWCore.ParameterSet.Config as cms

# CSC TF
from EventFilter.CSCTFRawToDigi.csctfpacker_cfi import *
# DT TF ???
from EventFilter.DTTFRawToDigi.dttfpacker_cfi import *
# RPC ???
# RCT/GCT
from EventFilter.GctRawToDigi.gctDigiToRaw_cfi import *
# GMT/GT
from EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi import *
L1DigiToRaw = cms.Sequence(csctfpacker+dttfpacker+gctDigiToRaw+l1GtPack)

