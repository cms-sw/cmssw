# The following comments couldn't be translated into the new config version:

# All/OuterSurface/InnerSurface/ImpactPoint/default(eff)
#

import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackEfficiencyClient_cfi import TrackEffClient

ClientTrackEfficiencyTkTracks = TrackEffClient.clone()
ClientTrackEfficiencyTkTracks.FolderName = 'Muons/TKTrack'
ClientTrackEfficiencyTkTracks.AlgoName = 'CTF'
ClientTrackEfficiencyTkTracks.trackEfficiency = False
