import FWCore.ParameterSet.Config as cms
CentralitypADQM = cms.EDAnalyzer("CentralitypADQM", 
                                 centralitycollection = cms.InputTag("pACentrality"),
                                 vertexcollection = cms.InputTag("offlinePrimaryVertices")
                                 )
