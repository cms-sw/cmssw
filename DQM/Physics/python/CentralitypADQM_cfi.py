import FWCore.ParameterSet.Config as cms
CentralitypADQM = DQMStep1Module('CentralitypADQM', 
                                 centralitycollection = cms.InputTag("pACentrality"),
                                 vertexcollection = cms.InputTag("offlinePrimaryVertices")
                                 )
