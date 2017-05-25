import FWCore.ParameterSet.Config as cms

rpcDataCertification = cms.EDProducer("RPCDataCertification",
                                      NumberOfEndcapDisks  = cms.untracked.int32(4)
                                      )

