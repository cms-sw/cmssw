import FWCore.ParameterSet.Config as cms

# DQM monitor module for BPhysics: onia resonances
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
bphysicsOniaDQM = DQMEDAnalyzer('BPhysicsOniaDQM',
                              MuonCollection = cms.InputTag("muons"),
                              vertex = cms.InputTag("offlinePrimaryVertices"),
                              lumiSummary = cms.InputTag("lumiProducer")
)


