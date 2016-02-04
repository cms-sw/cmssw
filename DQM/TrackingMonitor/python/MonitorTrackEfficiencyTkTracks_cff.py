# The following comments couldn't be translated into the new config version:

# All/OuterSurface/InnerSurface/ImpactPoint/default(track)
#

import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import TrackEffMon

MonitorTrackEfficiencyTkTracks = TrackEffMon.clone()
MonitorTrackEfficiencyTkTracks.TKTrackCollection = 'ctfWithMaterialTracksP5'
MonitorTrackEfficiencyTkTracks.STATrackCollection = 'cosmicMuons'
MonitorTrackEfficiencyTkTracks.FolderName = 'Muons/TKTrack'
MonitorTrackEfficiencyTkTracks.AlgoName = 'CTF'
MonitorTrackEfficiencyTkTracks.trackEfficiency = False
