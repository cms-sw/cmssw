import FWCore.ParameterSet.Config as cms

# DQM monitor module for BPhysics: onia resonances
bphysicsOniaDQM = DQMStep1Module('BPhysicsOniaDQM',
                              MuonCollection = cms.InputTag("muons"),
                              vertex = cms.InputTag("offlinePrimaryVertices"),
                              lumiSummary = cms.InputTag("lumiProducer")
)


