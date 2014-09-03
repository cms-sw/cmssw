import FWCore.ParameterSet.Config as cms

# import p+p collision sequences
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_cff import *

# SiStripMonitorTrack ####
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_hi  = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_hi.TrackProducer = "hiGlobalPrimTracks"
SiStripMonitorTrack_hi.Mod_On        = False

# TrackerMonitorTrack ####
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_hi = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_hi.Tracks              = 'hiGlobalPrimTracks'
MonitorTrackResiduals_hi.trajectoryInput     = "hiGlobalPrimTracks"
MonitorTrackResiduals_hi.OutputMEsInRootFile = False
MonitorTrackResiduals_hi.Mod_On              = False

# TrackingMonitor ####
import DQM.TrackingMonitor.TrackerHeavyIonTrackingMonitor_cfi
TrackMon_hi = DQM.TrackingMonitor.TrackerHeavyIonTrackingMonitor_cfi.TrackerHeavyIonTrackMon.clone()
TrackMon_hi.FolderName          = 'Tracking/TrackParameters'
TrackMon_hi.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'

#removed all modules using TkDetMap service
#SiStripDQMTier0_hi = cms.Sequence(APVPhases * consecutiveHEs * 
#                                  siStripFEDCheck * 
#                                  MonitorTrackResiduals_hi *
#                                  TrackMon_hi)
SiStripDQMTier0_hi = cms.Sequence(APVPhases * consecutiveHEs *
                                  siStripFEDCheck * siStripFEDMonitor *
                                  SiStripMonitorDigi * SiStripMonitorCluster *
                                  SiStripMonitorTrack_hi *
                                  MonitorTrackResiduals_hi *
                                  TrackMon_hi)
