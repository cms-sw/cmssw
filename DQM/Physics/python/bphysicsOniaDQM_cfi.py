import FWCore.ParameterSet.Config as cms

# DQM monitor module for BPhysics: onia resonances
oniaAnalyzer = cms.EDAnalyzer("BPhysicsOniaDQM",
                              MuonCollection = cms.InputTag("muons"),
)


