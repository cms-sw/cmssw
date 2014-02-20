import FWCore.ParameterSet.Config as cms

rpcDataCertification = cms.EDAnalyzer("RPCDataCertification",
                                      NumberOfEndcapDisks  = cms.untracked.int32(4)
                                      )

