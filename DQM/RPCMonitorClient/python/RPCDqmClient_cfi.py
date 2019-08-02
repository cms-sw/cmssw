import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcdqmclient = DQMEDHarvester("RPCDqmClient",                               
                              RPCDqmClientList = cms.untracked.vstring("RPCMultiplicityTest", "RPCDeadChannelTest", "RPCClusterSizeTest", "RPCOccupancyTest","RPCNoisyStripTest"),
                              DiagnosticPrescale = cms.untracked.int32(5),
                              #MinimumRPCEvents  = cms.untracked.int32(10000),
                              MinimumRPCEvents  = cms.untracked.int32(1),
                              RecHitTypeFolder = cms.untracked.string("AllHits"),
                              OfflineDQM = cms.untracked.bool(True),
                              UseRollInfo = cms.untracked.bool(False),
                              EnableRPCDqmClient  = cms.untracked.bool(True),
                              NumberOfEndcapDisks  = cms.untracked.int32(4),
                              NumberOfEndcapRings  = cms.untracked.int32(2)
                              )


rpcdqmMuonclient = DQMEDHarvester("RPCDqmClient",                               
                                  RPCDqmClientList = cms.untracked.vstring("RPCMultiplicityTest", "RPCDeadChannelTest", "RPCClusterSizeTest", "RPCOccupancyTest","RPCNoisyStripTest"),
                                  DiagnosticPrescale = cms.untracked.int32(5),
                                  #MinimumRPCEvents  = cms.untracked.int32(10000),
                                  MinimumRPCEvents  = cms.untracked.int32(1),
                                  RecHitTypeFolder = cms.untracked.string("Muon"),
                                  OfflineDQM = cms.untracked.bool(True),
                                  UseRollInfo = cms.untracked.bool(False),
                                  EnableRPCDqmClient  = cms.untracked.bool(True),
                                  NumberOfEndcapDisks  = cms.untracked.int32(4),
                                  NumberOfEndcapRings  = cms.untracked.int32(2)
                              )
