import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
rpcDcsInfo = DQMEDAnalyzer('RPCDcsInfo',
                            subSystemFolder = cms.untracked.string("RPC") ,
                            dcsInfoFolder = cms.untracked.string("DCSInfo") ,
                            ScalersRawToDigiLabel = cms.InputTag('scalersRawToDigi')
                            )
# foo bar baz
# 5N0gQnFv2WQdF
# qnAYLiyrsJA0F
