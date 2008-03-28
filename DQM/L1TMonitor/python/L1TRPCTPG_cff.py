import FWCore.ParameterSet.Config as cms

# include "EventFilter/RPCRawToDigi/data/RPCOrconCabling.cfi"
from EventFilter.RPCRawToDigi.RPCFrontierCabling_cfi import *
from DQM.L1TMonitor.L1TRPCTPG_cfi import *
rpcunpacker = cms.EDFilter("RPCUnpackingModule",
    InputLabel = cms.untracked.InputTag("source")
)

l1trpctpgpath = cms.Path(rpcunpacker*l1trpctpg)

