import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

rpcrechitprobabilityclient = DQMEDHarvester("RPCRecHitProbabilityClient",
                                            RPCFolder = cms.untracked.string("RPC"),
                                            GlobalFolder = cms.untracked.string("SummaryHistograms/RecHits"),
                                            MuonFolder = cms.untracked.string("Muon")
                                            )


