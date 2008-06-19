import FWCore.ParameterSet.Config as cms

import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
# SiStripMonitorTrack ####
# Cosmic Tracks
SiStripMonitorTrack_cosmicTk = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
# CKF Tracks
SiStripMonitorTrack_ckf = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
# Road Search  Tracks
SiStripMonitorTrack_rs = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# TrackerMonitorTrack ####
# Cosmic Track Finder
MonitorTrackResiduals_cosmicTk = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# CKF Tracks
MonitorTrackResiduals_ckf = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# Road Search  Tracks
MonitorTrackResiduals_rs = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
import DQM.TrackingMonitor.TrackingMonitor_cfi
# TrackingMonitor ####
# Cosmic Track Finder
TrackMon_cosmicTk = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
import DQM.TrackingMonitor.TrackingMonitor_cfi
# CKF Tracks
TrackMon_ckf = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
import DQM.TrackingMonitor.TrackingMonitor_cfi
# Road Search  Tracks
TrackMon_rs = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
SiStripDQMTier0_cosmicTk = cms.Sequence(SiStripMonitorTrack_cosmicTk*MonitorTrackResiduals_cosmicTk*TrackMon_cosmicTk)
SiStripDQMTier0_ckf = cms.Sequence(SiStripMonitorTrack_ckf*MonitorTrackResiduals_ckf*TrackMon_ckf)
SiStripDQMTier0_rs = cms.Sequence(SiStripMonitorTrack_rs*MonitorTrackResiduals_rs*TrackMon_rs)
SiStripDQMTier0 = cms.Sequence(SiStripMonitorTrack_cosmicTk*MonitorTrackResiduals_cosmicTk*TrackMon_cosmicTk*TrackMon_ckf*TrackMon_rs)
SiStripMonitorTrack_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
SiStripMonitorTrack_cosmicTk.Mod_On = False
SiStripMonitorTrack_cosmicTk.FolderName = 'SiStrip/Tracks'
SiStripMonitorTrack_ckf.TrackProducer = 'ctfWithMaterialTracksP5'
SiStripMonitorTrack_ckf.Mod_On = False
SiStripMonitorTrack_ckf.FolderName = 'SiStrip/Tracks'
SiStripMonitorTrack_rs.TrackProducer = 'rsWithMaterialTracksP5'
SiStripMonitorTrack_rs.Mod_On = False
SiStripMonitorTrack_rs.FolderName = 'SiStrip/Tracks'
# replace MonitorTrackResiduals_cosmicTk.Tracks = cosmictrackfinderP5 // This configurable is not read by the module code!
MonitorTrackResiduals_cosmicTk.trajectoryInput = 'cosmictrackfinderP5'
MonitorTrackResiduals_cosmicTk.OutputMEsInRootFile = False
MonitorTrackResiduals_cosmicTk.Mod_On = False
# replace MonitorTrackResiduals_ckf.Tracks = ctfWithMaterialTracksP5 // This configurable is not read by the module code!
MonitorTrackResiduals_ckf.trajectoryInput = 'ctfWithMaterialTracksP5'
MonitorTrackResiduals_ckf.OutputMEsInRootFile = False
MonitorTrackResiduals_ckf.Mod_On = False
# replace MonitorTrackResiduals_rs.Tracks = rsWithMaterialTracksP5 // This configurable is not read by the module code!
MonitorTrackResiduals_rs.trajectoryInput = 'rsWithMaterialTracksP5'
MonitorTrackResiduals_rs.OutputMEsInRootFile = False
MonitorTrackResiduals_rs.Mod_On = False
TrackMon_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
TrackMon_cosmicTk.AlgoName = 'CosmicTk'
TrackMon_cosmicTk.FolderName = 'SiStrip/Tracks'
TrackMon_cosmicTk.TkSizeMax = 25
TrackMon_cosmicTk.TkSizeBin = 25
TrackMon_ckf.TrackProducer = 'ctfWithMaterialTracksP5'
TrackMon_ckf.AlgoName = 'CKFTk'
TrackMon_ckf.FolderName = 'SiStrip/Tracks'
TrackMon_ckf.TkSizeMax = 25
TrackMon_ckf.TkSizeBin = 25
TrackMon_rs.TrackProducer = 'rsWithMaterialTracksP5'
TrackMon_rs.AlgoName = 'RSTk'
TrackMon_rs.FolderName = 'SiStrip/Tracks'
TrackMon_rs.TkSizeMax = 25
TrackMon_rs.TkSizeBin = 25

