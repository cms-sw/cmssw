import FWCore.ParameterSet.Config as cms

# SiStrip DQM Source

# Hardware Monitoring
from DQM.SiStripMonitorHardware.siStripFEDMonitor_P5_cff import *

# Condition DB Monitoring
from DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi import *

# DQMEventInfo
DqmEventInfoSiStrip = cms.EDAnalyzer( "DQMEventInfo",
    subSystemFolder = cms.untracked.string( 'SiStrip' )
)

# SiStripMonitoDigi
import DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi
SiStripMonitorDigiCAF                    = DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi.SiStripMonitorDigi.clone()
SiStripMonitorDigiCAF.SelectAllDetectors = True

# SiStripMonitorCluster
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
SiStripMonitorClusterCAF = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()
SiStripMonitorClusterCAF.SelectAllDetectors  = True
SiStripMonitorClusterCAF.StripQualityLabel   = ''

# SiStripMonitorTrack
# clone for cosmic track finder
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrackCAF_cosmicTk = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackCAF_cosmicTk.TrackProducer = 'cosmictrackfinderP5Refitter'
SiStripMonitorTrackCAF_cosmicTk.Mod_On        = True
# clone for CTF track finder
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrackCAF_ckf = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackCAF_ckf.TrackProducer = 'ctfWithMaterialTracksP5Refitter'
SiStripMonitorTrackCAF_ckf.Mod_On        = True
# clone for RS track finder
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrackCAF_rs = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackCAF_rs.TrackProducer = 'rsWithMaterialTracksP5Refitter'
SiStripMonitorTrackCAF_rs.Mod_On        = True

# TrackerMonitorTrack
# clone for cosmic track finder
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResidualsCAF_cosmicTk = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResidualsCAF_cosmicTk.Tracks              = 'cosmictrackfinderP5'
MonitorTrackResidualsCAF_cosmicTk.trajectoryInput     = 'cosmictrackfinderP5Refitter'
# clone for CTF track finder
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResidualsCAF_ckf = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResidualsCAF_ckf.Tracks              = 'ctfWithMaterialTracksP5'
MonitorTrackResidualsCAF_ckf.trajectoryInput     = 'ctfWithMaterialTracksP5Refitter'
# clone for RS track finder
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResidualsCAF_rs = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResidualsCAF_rs.Tracks              = 'rsWithMaterialTracksP5'
MonitorTrackResidualsCAF_rs.trajectoryInput     = 'rsWithMaterialTracksP5Refitter'

# TrackingMonitor
# clone for cosmic track finder
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMonCAF_cosmicTk = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMonCAF_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
TrackMonCAF_cosmicTk.AlgoName      = 'CosmicTk'
TrackMonCAF_cosmicTk.FolderName    = 'SiStrip/Tracks'
# clone for CTF track finder
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMonCAF_ckf = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMonCAF_ckf.TrackProducer = 'ctfWithMaterialTracksP5'
TrackMonCAF_ckf.AlgoName      = 'CKFTk'
TrackMonCAF_ckf.FolderName    = 'SiStrip/Tracks'
# clone for RS track finder
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMonCAF_rs = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMonCAF_rs.TrackProducer = 'rsWithMaterialTracksP5'
TrackMonCAF_rs.AlgoName      = 'RSTk'
TrackMonCAF_rs.FolderName    = 'SiStrip/Tracks'

# Scheduling
SiStripDQMSourceGlobalRunCAF_fromRAW  = cms.Sequence( siStripFEDMonitor )
SiStripDQMSourceGlobalRunCAF_common   = cms.Sequence( CondDataMonitoring + DqmEventInfoSiStrip + SiStripMonitorDigiCAF + SiStripMonitorClusterCAF )
SiStripDQMSourceGlobalRunCAF_cosmikTk = cms.Sequence( SiStripMonitorTrackCAF_cosmicTk + MonitorTrackResidualsCAF_cosmicTk + TrackMonCAF_cosmicTk )
SiStripDQMSourceGlobalRunCAF_ckf      = cms.Sequence( SiStripMonitorTrackCAF_ckf      + MonitorTrackResidualsCAF_ckf      + TrackMonCAF_ckf )
SiStripDQMSourceGlobalRunCAF_rs       = cms.Sequence( SiStripMonitorTrackCAF_rs       + MonitorTrackResidualsCAF_rs       + TrackMonCAF_rs )
SiStripDQMSourceGlobalRunCAF          = cms.Sequence( SiStripDQMSourceGlobalRunCAF_common + SiStripDQMSourceGlobalRunCAF_cosmikTk + SiStripDQMSourceGlobalRunCAF_ckf + SiStripDQMSourceGlobalRunCAF_rs )
SiStripDQMSourceGlobalRunCAF_reduced  = cms.Sequence( SiStripDQMSourceGlobalRunCAF_common + TrackMonCAF_cosmicTk                  + SiStripDQMSourceGlobalRunCAF_ckf + TrackMonCAF_rs )


