# The following comments couldn't be translated into the new config version:

# All/OuterSurface/InnerSurface/ImpactPoint/default(track)
#

import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi import TrackEffMon

MonitorTrackEfficiencySTACosmicMuons = TrackEffMon.clone()
MonitorTrackEfficiencySTACosmicMuons.TKTrackCollection = 'ctfWithMaterialTracksP5'
MonitorTrackEfficiencySTACosmicMuons.STATrackCollection = 'cosmicMuons'
MonitorTrackEfficiencySTACosmicMuons.FolderName = 'Muons/cosmicMuons'
MonitorTrackEfficiencySTACosmicMuons.AlgoName = 'STA'
    
    
