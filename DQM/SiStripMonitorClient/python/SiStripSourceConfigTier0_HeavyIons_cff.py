import FWCore.ParameterSet.Config as cms

# import p+p collision sequences
from DQM.SiStripMonitorClient.SiStripSourceConfigTier0_cff import *

# SiStripMonitorTrack ####
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_hi  = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_hi.TrackProducer = "hiGeneralTracks"
SiStripMonitorTrack_hi.Mod_On        = False

# TrackerMonitorTrack ####
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_hi = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_hi.Tracks              = 'hiGeneralTracks'
MonitorTrackResiduals_hi.trajectoryInput     = "hiGeneralTracks"
MonitorTrackResiduals_hi.Mod_On              = False

SiStripDQMTier0_hi = cms.Sequence(APVPhases * consecutiveHEs *
                                  siStripFEDCheck * siStripFEDMonitor *
                                  SiStripMonitorDigi * SiStripMonitorCluster *
                                  SiStripMonitorTrack_hi *
                                  MonitorTrackResiduals_hi)
