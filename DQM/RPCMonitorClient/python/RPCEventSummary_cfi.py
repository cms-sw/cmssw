import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcEventSummary = DQMEDHarvester("RPCEventSummary",
                                 EventInfoPath = cms.untracked.string('RPC/EventInfo'),
                                 PrescaleFactor = cms.untracked.int32(5),
                                 MinimumRPCEvents = cms.untracked.int32(10000),
                                 NumberOfEndcapDisks = cms.untracked.int32(4),
                                 EnableEndcapSummary = cms.untracked.bool(True),
                                 OfflineDQM = cms.untracked.bool(True),
                                 RecHitTypeFolder = cms.untracked.string("AllHits")
                                 )
