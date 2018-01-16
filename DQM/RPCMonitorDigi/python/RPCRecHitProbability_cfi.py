import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
rpcrechitprobability = DQMEDAnalyzer('RPCRecHitProbability',
                                      SaveRootFile = cms.untracked.bool(False),
                                      RootFileName = cms.untracked.string('RPCRecHitProbabilityDQM.root'),
                                      MuonPtCut = cms.untracked.double(3.0),
                                      MuonEtaCut= cms.untracked.double(1.9),
                                      ScalersRawToDigiLabel = cms.InputTag('scalersRawToDigi'),
                                      MuonLabel =  cms.InputTag('muons'),
                                      RPCFolder = cms.untracked.string('RPC'),
                                      GlobalFolder = cms.untracked.string('SummaryHistograms/RecHits'),
                                      MuonFolder = cms.untracked.string("Muon")
                                      )


