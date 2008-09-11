import FWCore.ParameterSet.Config as cms

# SiStripMonitorTrack ####
# Clone for Cosmic Tracks
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_cosmicTk = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
SiStripMonitorTrack_cosmicTk.Mod_On = False
SiStripMonitorTrack_cosmicTk.FolderName = 'SiStrip/Tracks'
# Clone for CKF Tracks
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_ckf = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_ckf.TrackProducer = 'ctfWithMaterialTracksP5'
SiStripMonitorTrack_ckf.Mod_On = False
SiStripMonitorTrack_ckf.FolderName = 'SiStrip/Tracks'
# Clone for Road Search  Tracks
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_rs = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_rs.TrackProducer = 'rsWithMaterialTracksP5'
SiStripMonitorTrack_rs.Mod_On = False
SiStripMonitorTrack_rs.FolderName = 'SiStrip/Tracks'

# TrackerMonitorTrack ####
# Clone for Cosmic Track Finder
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_cosmicTk = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_cosmicTk.trajectoryInput = 'cosmictrackfinderP5'
MonitorTrackResiduals_cosmicTk.OutputMEsInRootFile = False
MonitorTrackResiduals_cosmicTk.Mod_On = False
# Clone for CKF Tracks
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_ckf = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_ckf.trajectoryInput = 'ctfWithMaterialTracksP5'
MonitorTrackResiduals_ckf.OutputMEsInRootFile = False
MonitorTrackResiduals_ckf.Mod_On = False
# Clone for Road Search  Tracks
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_rs = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_rs.trajectoryInput = 'rsWithMaterialTracksP5'
MonitorTrackResiduals_rs.OutputMEsInRootFile = False
MonitorTrackResiduals_rs.Mod_On = False

# TrackingMonitor ####
# Clone for Cosmic Track Finder
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_cosmicTk = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
TrackMon_cosmicTk.AlgoName = 'CosmicTk'
TrackMon_cosmicTk.FolderName = 'SiStrip/Tracks'

# Clone for CKF Tracks
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_ckf = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_ckf.TrackProducer = 'ctfWithMaterialTracksP5'
TrackMon_ckf.AlgoName = 'CKFTk'
TrackMon_ckf.FolderName = 'SiStrip/Tracks'

# Clone for Road Search  Tracks
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_rs = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_rs.TrackProducer = 'rsWithMaterialTracksP5'
TrackMon_rs.AlgoName = 'RSTk'
TrackMon_rs.FolderName = 'SiStrip/Tracks'

# Clone for Beam Halo Muon Tracks
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_bhmuon = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_bhmuon.TrackProducer = 'ctfWithMaterialTracksBeamHaloMuon'
TrackMon_bhmuon.AlgoName = 'BHMuonTk'
TrackMon_bhmuon.FolderName = 'SiStrip/Tracks'


# Sequences 
SiStripDQMTier0_cosmicTk = cms.Sequence(SiStripMonitorTrack_cosmicTk*MonitorTrackResiduals_cosmicTk*TrackMon_cosmicTk)

SiStripDQMTier0_ckf = cms.Sequence(SiStripMonitorTrack_ckf*MonitorTrackResiduals_ckf*TrackMon_ckf)

SiStripDQMTier0_rs = cms.Sequence(SiStripMonitorTrack_rs*MonitorTrackResiduals_rs*TrackMon_rs)

SiStripDQMTier0 = cms.Sequence(SiStripMonitorTrack_ckf*MonitorTrackResiduals_ckf*TrackMon_cosmicTk*TrackMon_ckf*TrackMon_rs*TrackMon_bhmuon)


