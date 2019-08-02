import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcChamberQuality = DQMEDHarvester("RPCChamberQuality",
                                   OfflineDQM = cms.untracked.bool(True),
                                   PrescaleFactor  = cms.untracked.int32(5),
                                   NumberOfEndcapDisks  = cms.untracked.int32(4),
                                   #MinimumRPCEvents = cms.untracked.int32(10000),
                                   MinimumRPCEvents = cms.untracked.int32(1),
                                   RecHitTypeFolder = cms.untracked.string("AllHits")
                                   )


rpcMuonChamberQuality = DQMEDHarvester("RPCChamberQuality",
                                       OfflineDQM = cms.untracked.bool(True),
                                       PrescaleFactor  = cms.untracked.int32(5),
                                       NumberOfEndcapDisks  = cms.untracked.int32(4),
                                       #MinimumRPCEvents = cms.untracked.int32(10000),
                                       MinimumRPCEvents = cms.untracked.int32(1),
                                       RecHitTypeFolder = cms.untracked.string("Muon")
                                       )
