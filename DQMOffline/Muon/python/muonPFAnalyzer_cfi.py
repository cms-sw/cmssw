import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
muonPFsequence = DQMEDAnalyzer('MuonPFAnalyzer',
                                inputTagMuonReco     = cms.InputTag("muons"),
                                inputTagGenParticles = cms.InputTag("genParticles"),
                                inputTagVertex       = cms.InputTag("offlinePrimaryVertices"),
                                inputTagBeamSpot     = cms.InputTag("offlineBeamSpot"),
                                
                                runOnMC = cms.bool(False),
                                folder = cms.string("Muons/MuonPFAnalyzer/"),
                                
                                recoGenDeltaR   = cms.double(0.1),
                                relCombIsoCut   = cms.double(0.15),
                                highPtThreshold = cms.double(200.)
                                )



