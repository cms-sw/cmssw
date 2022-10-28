import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
DiMuonVertexMonitor = DQMEDAnalyzer('DiMuonVertexMonitor',
                                    muonTracks = cms.InputTag('ALCARECOTkAlDiMuon'),
                                    decayMotherName = cms.string('Z'),
                                    vertices = cms.InputTag('offlinePrimaryVertices'),
                                    FolderName = cms.string('DiMuonVertexMonitor'),
                                    maxSVdist = cms.double(50))
