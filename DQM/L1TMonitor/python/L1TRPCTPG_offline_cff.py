import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TRPCTPG_cfi import *

from EventFilter.RPCRawToDigi.RPCSQLiteCabling_cfi import *

from EventFilter.RPCRawToDigi.rpcUnpacker_cfi import *

#l1trpctpg.rpctpgSource = cms.InputTag("rpcunpacker")
#l1trpctpg.rpctfSource = cms.InputTag("gtUnpack")


l1trpctpgpath = cms.Path(rpcunpacker*l1trpctpg)

