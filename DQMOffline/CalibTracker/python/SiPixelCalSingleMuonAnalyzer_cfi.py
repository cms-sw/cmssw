import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
siPixelCalSingleMuonAnalyzer = DQMEDAnalyzer('SiPixelCalSingleMuonAnalyzer',
                                             clusterCollection = cms.InputTag('ALCARECOSiPixelCalSingleMuonTight'),
                                             nearByClusterCollection = cms.InputTag('closebyPixelClusters'),
                                             trajectoryInput = cms.InputTag('refittedTracks'),
                                             muonTracks = cms.InputTag('ALCARECOSiPixelCalSingleMuonTight'),
                                             distToTrack = cms.InputTag('trackDistances'),
                                             dqmPath =  cms.string("SiPixelCalSingleMuonTight"),
                                             skimmedGeometryPath = cms.string('SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt'))
