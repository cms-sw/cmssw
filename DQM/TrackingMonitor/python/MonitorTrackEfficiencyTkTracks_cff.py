# The following comments couldn't be translated into the new config version:

# All/OuterSurface/InnerSurface/ImpactPoint/default(track)
#

import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import TrackEffMon

MonitorTrackEfficiencyTkTracks = TrackEffMon.clone(
    TKTrackCollection = 'ctfWithMaterialTracksP5',
    STATrackCollection = 'cosmicMuons',
    FolderName = 'Muons/TKTrack',
    AlgoName = 'CTF',
    trackEfficiency = False
)
# foo bar baz
# 7vOXR7WX9ASIx
# C2F2BPhoV4tDP
