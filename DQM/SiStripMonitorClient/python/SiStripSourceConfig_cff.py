import FWCore.ParameterSet.Config as cms

# Hardware Monitor ###
from DQM.SiStripMonitorHardware.siStripFEDMonitor_P5_cff import *

# Pedestal Monitor ###
from DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi import *
PedsMon.StripQualityLabel = ''
PedsMon.RunTypeFlag = 'CalculatedPlotsOnly'

# Digi Monitor #####
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi import *
SiStripMonitorDigi.SelectAllDetectors = True

# Cluster Monitor ####
from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *
SiStripMonitorCluster.SelectAllDetectors = True
SiStripMonitorCluster.StripQualityLabel = ''

# On/Off Track Cluster Monitor ####
# Clone for Sim data
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrackSim = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackSim.TrackProducer = 'TrackRefitter'
SiStripMonitorTrackSim.TrackLabel    = ''
SiStripMonitorTrackSim.Cluster_src   = 'siStripClusters'
SiStripMonitorTrackSim.Mod_On        = True

# Clone for Real Data
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrackReal = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackReal.TrackProducer = 'ctfWithMaterialTracksP5'
SiStripMonitorTrackReal.TrackLabel    = ''
SiStripMonitorTrackReal.Cluster_src   = 'siStripClusters'
SiStripMonitorTrackReal.Mod_On        = True

# Clone for Real Data (Collision)
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrackColl = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrackColl.TrackProducer = 'generalTracks'
SiStripMonitorTrackColl.TrackLabel    = ''
SiStripMonitorTrackColl.Cluster_src   = 'siStripClusters'
SiStripMonitorTrackColl.Mod_On        = True


# Residual Monitor ####
# Clone for Sim Data
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResidualsSim = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
# Clone for Real Data
MonitorTrackResidualsReal = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResidualsReal.Tracks              = 'ctfWithMaterialTracksP5'
MonitorTrackResidualsReal.trajectoryInput     = 'ctfWithMaterialTracksP5'
# Clone for Real Data
MonitorTrackResidualsColl = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResidualsColl.Tracks              = 'generalTracks'
MonitorTrackResidualsColl.trajectoryInput     = 'generalTracks'


# Tracking Monitor ####
# Clone for Sim Data
import DQM.TrackingMonitor.TrackingMonitor_cfi
TrackMonSim = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
TrackMonSim.FolderName = 'Tracking/TrackParameters'
# Clone for Real Data (Cosmic)
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMonReal = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMonReal.TrackProducer = 'ctfWithMaterialTracksP5'
TrackMonReal.FolderName = 'Tracking/TrackParameters'
TrackMonReal.AlgoName = 'CKFTk'
TrackMonReal.doSeedParameterHistos = True

# Clone for Real Data (Collison)
import DQM.TrackingMonitor.TrackingMonitor_cfi
TrackMonColl = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
TrackMonColl.TrackProducer = 'generalTracks'
TrackMonColl.FolderName = 'Tracking/TrackParameters'
TrackMonColl.AlgoName = 'CKFTk'
TrackMonColl.doSeedParameterHistos = True
# Sequences
#removed modules using TkDetMap service
#SiStripSourcesSimData = cms.Sequence(SiStripMonitorTrackSim*MonitorTrackResidualsSim*TrackMonSim)
#SiStripSourcesRealData = cms.Sequence(SiStripMonitorTrackReal*MonitorTrackResidualsReal*TrackMonReal)
#SiStripSourcesRealDataCollision = cms.Sequence(SiStripMonitorTrackColl*MonitorTrackResidualsColl*TrackMonColl)
SiStripSourcesSimData = cms.Sequence(SiStripMonitorDigi*SiStripMonitorCluster*SiStripMonitorTrackSim*MonitorTrackResidualsSim*TrackMonSim)
SiStripSourcesRealData = cms.Sequence(SiStripMonitorDigi*SiStripMonitorCluster*SiStripMonitorTrackReal*MonitorTrackResidualsReal*TrackMonReal)
SiStripSourcesRealDataCollision = cms.Sequence(SiStripMonitorDigi*SiStripMonitorCluster*SiStripMonitorTrackColl*MonitorTrackResidualsColl*TrackMonColl)




