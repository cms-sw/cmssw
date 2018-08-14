import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcDCSSummary = DQMEDHarvester("RPCDCSSummary", 
                               NumberOfEndcapDisks  = cms.untracked.int32(4),
                               )
