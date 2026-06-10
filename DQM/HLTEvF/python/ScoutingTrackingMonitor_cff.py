import FWCore.ParameterSet.Config as cms

## See DQMOffline/HLTScouting/python/HLTScoutingDqmOffline_cff.py
from HLTriggerOffline.Scouting.ScoutingTrackMonitor_cfi import  ScoutingTrackMonitor
from HLTriggerOffline.Scouting.RecoTrackFromScoutingMonitor_cfi import scoutingRecoTrackMonitor as _scoutingRecoTrackMonitor
from HLTriggerOffline.Scouting.RecoTrackFromScoutingMonitor_cff import recoTracksFromScouting,recoVerticesFromScouting

ScoutingTrackMonitorOnline = ScoutingTrackMonitor.clone(
    topFolderName = 'HLT/ScoutingOnline/Tracks'
)

ScoutingRecoTrackMonitorOnline = _scoutingRecoTrackMonitor.clone(
    doLumiAnalysis = False, # does not play nice with the online saver
    FolderName     = 'HLT/ScoutingOnline/Tracks',
    BSFolderName   = 'HLT/ScoutingOnline/Tracks',
    PVFolderName   = 'HLT/ScoutingOnline/Tracks'
)

ScoutingTracksMonitoring = cms.Sequence(ScoutingTrackMonitorOnline *
                                        recoTracksFromScouting *
                                        recoVerticesFromScouting *
                                        ScoutingRecoTrackMonitorOnline)
