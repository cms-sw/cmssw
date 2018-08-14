import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcDaqInfo = DQMEDHarvester("RPCDaqInfo",
                            NumberOfEndcapDisks  = cms.untracked.int32(4)
                            )

