import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TRPCTPG_cfi import *

from EventFilter.RPCRawToDigi.RPCFrontierCabling_cfi import *

from EventFilter.RPCRawToDigi.rpcUnpacker_cfi import *

#    include "EventFilter/RPCRawToDigi/data/RPCFrontierCabling.cfi"
rpcunpacker.InputLabel = cms.untracked.InputTag("rawDataCollector")

l1trpctpgpath = cms.Path(rpcunpacker*l1trpctpg)

