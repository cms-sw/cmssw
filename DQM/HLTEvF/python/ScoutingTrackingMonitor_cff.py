import FWCore.ParameterSet.Config as cms

## See DQMOffline/HLTScouting/python/HLTScoutingDqmOffline_cff.py
from HLTriggerOffline.Scouting.ScoutingTrackMonitor_cfi import *

ScoutingTrackMonitorOnline = ScoutingTrackMonitor.clone(
    topFolderName = 'HLT/ScoutingOnline/Tracks'
)

ScoutingTracksMonitoring = cms.Sequence(ScoutingTrackMonitorOnline)
