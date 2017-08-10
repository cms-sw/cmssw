import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcEfficiencyShiftHisto = DQMEDHarvester("RPCEfficiencyShiftHisto",
   EffCut = cms.untracked.int32(90),
   GlobalFolder = cms.untracked.string('RPC/RPCEfficiency/'),
   NumberOfEndcapDisks = cms.untracked.int32(4)
)
