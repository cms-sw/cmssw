import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
CentralitypADQM = DQMEDAnalyzer('CentralitypADQM', 
                                 centralitycollection = cms.InputTag("pACentrality"),
                                 vertexcollection = cms.InputTag("offlinePrimaryVertices")
                                 )
