import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
rpcTTUMonitor = DQMEDAnalyzer('RPCTTUMonitor',
                               TTUFolder =cms.string("RPC/TTU"),
                               OutPutFile = cms.string(""),
                               GTReadoutRcd     = cms.InputTag("gtDigis"),
                               GMTReadoutRcd    = cms.InputTag("gtDigis" ),
                               L1TTEmuBitsLabel = cms.InputTag("rpcTechnicalTrigger"),
                               BitNumbers       = cms.vuint32(24,25,26,27,28,29,30) )


