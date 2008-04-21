import FWCore.ParameterSet.Config as cms

import copy
from EventFilter.RPCRawToDigi.rpcUnpacker_cfi import *
muonRPCDigis = copy.deepcopy(rpcunpacker)
RPCRawToDigi = cms.Sequence(muonRPCDigis)
muonRPCDigis.InputLabel = 'rawDataCollector'

