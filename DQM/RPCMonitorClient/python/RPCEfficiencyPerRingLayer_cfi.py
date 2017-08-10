import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcEfficiencyPerRingLayer = DQMEDHarvester("RPCEfficiencyPerRingLayer",
                                           GlobalFolder = cms.untracked.string('RPC/RPCEfficiency/'),
                                           NumberOfEndcapDisks  = cms.untracked.int32(4),
                                           NumberOfInnermostEndcapRings  = cms.untracked.int32(2)
                                           )
