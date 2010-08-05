import FWCore.ParameterSet.Config as cms

rpcnoiseclient = cms.EDAnalyzer("RPCDqmClient",
                              EnableRPCDqmClients=cms.untracked.bool(True),
                              RPCDqmClientList = cms.untracked.vstring("RPCMultiplicityTest","RPCDeadChannelTest","RPCClusterSizeTest","RPCOccupancyTest","RPCNoisyStripTest"),
                              DiagnosticPrescale = cms.untracked.int32(5),
                              RPCFolder =cms.untracked.string("RPC"),
                              NoiseOrMuons =cms.untracked.string("Noise"),
                              GlobalFolder =cms.untracked.string("SummaryHistograms"),
                              MinimumRPCEvents  = cms.untracked.int32(10000),
                              NumberOfEndcapRings  = cms.untracked.int32(2),
                              NumberOfEndcapDisks  = cms.untracked.int32(3)
                              )

rpcmuonclient = cms.EDAnalyzer("RPCDqmClient",
                              EnableRPCDqmClients=cms.untracked.bool(True),
                              RPCDqmClientList = cms.untracked.vstring("RPCMultiplicityTest","RPCDeadChannelTest","RPCClusterSizeTest","RPCOccupancyTest","RPCNoisyStripTest"),
                              DiagnosticPrescale = cms.untracked.int32(5),
                              RPCFolder =cms.untracked.string("RPC"),
                              NoiseOrMuons =cms.untracked.string("Muon"),
                              GlobalFolder =cms.untracked.string("SummaryHistograms"),
                              MinimumRPCEvents  = cms.untracked.int32(10000),
                              NumberOfEndcapRings  = cms.untracked.int32(2),
                              NumberOfEndcapDisks  = cms.untracked.int32(3)
                              )
