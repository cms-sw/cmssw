import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
rpcFEDIntegrity = DQMEDAnalyzer('RPCFEDIntegrity',
                                 RPCPrefixDir =  cms.untracked.string('RPC/FEDIntegrity'),
                                 RPCRawCountsInputTag = cms.untracked.InputTag('muonRPCDigis'),
                                 NumberOfFED = cms.untracked.int32(3)
                                 )


