import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorHardware.siStripFEDMonitor_P5_cff import *

# SiStripMonitorDigi ####
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi import *
SiStripMonitorDigi.SelectAllDetectors = True
SiStripMonitorDigi.TProfTotalNumberOfDigis.subdetswitchon = True
SiStripMonitorDigi.TProfDigiApvCycle.subdetswitchon = True

# SiStripMonitorCluster ####
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
SiStripMonitorClusterReal = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()
SiStripMonitorClusterReal.OutputMEsInRootFile = False
SiStripMonitorClusterReal.SelectAllDetectors = True
SiStripMonitorClusterReal.TProfTotalNumberOfClusters.subdetswitchon = True
SiStripMonitorClusterReal.TProfClustersApvCycle.subdetswitchon = True

# SiStripMonitorTrack ####
# Clone for Cosmic Track Finder
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_cosmicTk = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
SiStripMonitorTrack_cosmicTk.FolderName    = 'Tracking/TrackParameters'
SiStripMonitorTrack_cosmicTk.Mod_On        = True

# Clone for CKF Tracks
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_ckf = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_ckf.TrackProducer      = 'ctfWithMaterialTracksP5'
SiStripMonitorTrack_ckf.FolderName         = 'Tracking/TrackParameters'
SiStripMonitorTrack_ckf.Mod_On             = True

# Clone fir Road Search  Tracks
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_rs = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_rs.TrackProducer       = 'rsWithMaterialTracksP5'
SiStripMonitorTrack_rs.FolderName          = 'Tracking/TrackParameters'
SiStripMonitorTrack_rs.Mod_On              = True

# Clone for General Tracks (for Collision)
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_gentk = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_gentk.TrackProducer    = 'generalTracks'
SiStripMonitorTrack_gentk.FolderName       = 'Tracking/TrackParameters'
SiStripMonitorTrack_gentk.Mod_On           = True

# TrackerMonitorTrack ####
# Clone for Cosmic Track Finder
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_cosmicTk = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_cosmicTk.Tracks              = 'cosmictrackfinderP5'
MonitorTrackResiduals_cosmicTk.trajectoryInput     = 'cosmictrackfinderP5'
MonitorTrackResiduals_cosmicTk.OutputMEsInRootFile = False
MonitorTrackResiduals_cosmicTk.Mod_On              = False

# Clone for CKF Tracks
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_ckf = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_ckf.Tracks                   = 'ctfWithMaterialTracksP5'
MonitorTrackResiduals_ckf.trajectoryInput          = 'ctfWithMaterialTracksP5'
MonitorTrackResiduals_ckf.OutputMEsInRootFile      = False
MonitorTrackResiduals_ckf.Mod_On                   = False

# Clone for Road Search  Tracks
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_rs = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_rs.Tracks                    = 'rsWithMaterialTracksP5'
MonitorTrackResiduals_rs.trajectoryInput           = 'rsWithMaterialTracksP5'
MonitorTrackResiduals_rs.OutputMEsInRootFile       = False
MonitorTrackResiduals_rs.Mod_On                    = False

# Clone for General Track (for Collision data)
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_gentk = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_gentk.Tracks                 = 'generalTracks'
MonitorTrackResiduals_gentk.trajectoryInput        = 'generalTracks'
MonitorTrackResiduals_gentk.OutputMEsInRootFile    = False
MonitorTrackResiduals_gentk.Mod_On                 = False

# TrackingMonitor ####
# Clone for Cosmic Track Finder
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_cosmicTk = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
TrackMon_cosmicTk.AlgoName      = 'CosmicTk'
TrackMon_cosmicTk.FolderName    = 'Tracking/TrackParameters'
TrackMon_cosmicTk.doSeedParameterHistos = True

# Clone for CKF Tracks
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_ckf = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_ckf.TrackProducer      = 'ctfWithMaterialTracksP5'
TrackMon_ckf.AlgoName           = 'GenTk'
TrackMon_ckf.FolderName         = 'Tracking/TrackParameters'
TrackMon_ckf.doSeedParameterHistos = True

# Clone for Road Search  Tracks
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_rs = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_rs.TrackProducer       = 'rsWithMaterialTracksP5'
TrackMon_rs.AlgoName            = 'RSTk'
TrackMon_rs.FolderName          = 'Tracking/TrackParameters'
TrackMon_rs.doSeedParameterHistos = True

# Clone for General Track (for Collision data)

import DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi
TrackMon_gentk = DQM.TrackingMonitor.TrackerCollisionTrackingMonitor_cfi.TrackerCollisionTrackMon.clone()
TrackMon_gentk.FolderName          = 'Tracking/TrackParameters'
TrackMon_gentk.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'

# Tracking Efficiency
# Clone for Cosmic Tracks
import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
TrackEffMon_cosmicTk = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
TrackEffMon_cosmicTk.TKTrackCollection             = 'cosmictrackfinderP5'
TrackEffMon_cosmicTk.AlgoName                      = 'CosmicTk'
TrackEffMon_cosmicTk.FolderName                    = 'Tracking/TrackParameters/TrackEfficiency'

# Clone for CKF Tracks
import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
TrackEffMon_ckf = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
TrackEffMon_ckf.TKTrackCollection                  = 'ctfWithMaterialTracksP5'
TrackEffMon_ckf.AlgoName                           = 'CKFTk'
TrackEffMon_ckf.FolderName                         = 'Tracking/TrackParameters/TrackEfficiency'

# Clone for RS Tracks
import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
TrackEffMon_rs = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
TrackEffMon_rs.TKTrackCollection                   = 'rsWithMaterialTracksP5'
TrackEffMon_rs.AlgoName                            = 'RSTk'
TrackEffMon_rs.FolderName                          = 'Tracking/TrackParameters/TrackEfficiency'

# Clone for Beam Halo  Tracks
import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
TrackEffMon_bhmuon = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
TrackEffMon_bhmuon.TKTrackCollection               = 'ctfWithMaterialTracksBeamHaloMuon'
TrackEffMon_bhmuon.AlgoName                        = 'BHMuonTk'
TrackEffMon_bhmuon.FolderName                      = 'Tracking/TrackParameters/TrackEfficiency'

# Services needed for TkHistoMap
TkDetMap = cms.Service("TkDetMap")
SiStripDetInfoFileReade = cms.Service("SiStripDetInfoFileReader")
