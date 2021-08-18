import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcDcsInfoClient = DQMEDHarvester("RPCDcsInfoClient",
                                  dcsInfoFolder = cms.untracked.string("RPC/DCSInfo"),
                                  dqmProvInfoFolder = cms.untracked.string("Info/EventInfo")
                                  )
