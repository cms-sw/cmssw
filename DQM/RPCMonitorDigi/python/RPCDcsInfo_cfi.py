import FWCore.ParameterSet.Config as cms

rpcDcsInfo = cms.EDAnalyzer("RPCDcsInfo",
                            subSystemFolder = cms.untracked.string("RPC") ,
                            dcsInfoFolder = cms.untracked.string("DCSInfo") ,
                            ScalersRawToDigiLabel = cms.untracked.string("scalersRawToDigi")
                            )
