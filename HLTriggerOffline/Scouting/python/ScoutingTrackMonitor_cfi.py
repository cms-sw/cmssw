import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ScoutingTrackMonitor =  DQMEDAnalyzer('ScoutingTrackMonitor',
                                      tracks = cms.InputTag('hltScoutingTrackPacker'),
                                      vertices = cms.InputTag('hltScoutingPrimaryVertexPacker', 'primaryVtx'),
                                      topFolderName = cms.string('HLT/ScoutingOffline/Tracks')
                                      )
