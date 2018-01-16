import FWCore.ParameterSet.Config as cms

rpcDcsInfo = DQMStep1Module('RPCDcsInfo',
                            subSystemFolder = cms.untracked.string("RPC") ,
                            dcsInfoFolder = cms.untracked.string("DCSInfo") ,
                            ScalersRawToDigiLabel = cms.InputTag('scalersRawToDigi')
                            )
