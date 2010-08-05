import FWCore.ParameterSet.Config as cms

rpcChamberQuality = cms.EDAnalyzer("RPCChamberQuality",
                                   EnableRPCDqmClients = cms.untracked.bool(True),
                                   DiagnosticPrescale = cms.untracked.int32(5),
                                   RPCFolder = cms.untracked.string("RPC"),
                                   NoiseOrMuons = cms.untracked.string("Noise"),
                                   GlobalFolder = cms.untracked.string("SummaryHistograms"),
                                   MinimumRPCEvents  = cms.untracked.int32(10000),
                                   NumberOfEndcapDisks  = cms.untracked.int32(3)
                                   )
