import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ScoutingDiMuonVertexMonitor = DQMEDAnalyzer("ScoutingDiMuonVertexMonitor",
                                            FolderName = cms.string('HLT/ScoutingOffline/DiMuon'),
                                            decayMotherName = cms.string('Z'),
                                            muons = cms.InputTag('hltScoutingMuonPackerVtx'),
                                            primaryVertices = cms.InputTag('hltScoutingPrimaryVertexPacker', 'primaryVtx'),
                                            secondaryVertices = cms.InputTag('hltScoutingMuonPackerVtx', 'displacedVtx'),
                                            applyMuonID = cms.bool(True),
                                            minMuonPt = cms.double(3),
                                            maxMuonEta = cms.double(2.4),
                                            minVtxProb = cms.double(0.005),
                                            maxSVdistXY = cms.double(50))
