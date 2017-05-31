import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcDataCertification = DQMEDHarvester("RPCDataCertification",
                                      NumberOfEndcapDisks  = cms.untracked.int32(4)
                                      )

