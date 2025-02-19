import FWCore.ParameterSet.Config as cms

rpcrechitprobabilityclient = cms.EDAnalyzer("RPCRecHitProbabilityClient",
                                            RPCFolder = cms.untracked.string("RPC"),
                                            GlobalFolder = cms.untracked.string("SummaryHistograms/RecHits"),
                                            MuonFolder = cms.untracked.string("Muon")
                                            )


