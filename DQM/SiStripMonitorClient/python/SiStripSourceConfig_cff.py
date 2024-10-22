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

# refitter ### (FIXME rename, move)
from DQM.SiPixelMonitorTrack.RefitterForPixelDQM import *

# On/Off Track Cluster Monitor ####
# Clone for Sim dat
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrackSim = SiStripMonitorTrack.clone(
    TrackProducer = 'TrackRefitter',
    TrackLabel = '',
    Cluster_src = 'siStripClusters',
    Mod_On = True
)

# Clone for Real Dat
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrackReal = SiStripMonitorTrack.clone(
    TrackProducer = 'ctfWithMaterialTracksP5',
    TrackLabel = '',
    Cluster_src = 'siStripClusters',
    Mod_On = True
)

# Clone for Real Data (Collision
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrackColl = SiStripMonitorTrack.clone(
    TrackProducer = 'refittedForPixelDQM',
    TrackLabel = '',
    Cluster_src = 'siStripClusters',
    Mod_On = True
)

# Residual Monitor ####
# Clone for Sim Data
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResidualsSim = MonitorTrackResiduals.clone()
# Clone for Real Data
MonitorTrackResidualsReal = MonitorTrackResiduals.clone(
    Tracks = 'ctfWithMaterialTracksP5',
    trajectoryInput = 'ctfWithMaterialTracksP5'
)
# Clone for Real Data
MonitorTrackResidualsColl = MonitorTrackResiduals.clone(
    Tracks = 'refittedForPixelDQM',
    trajectoryInput = 'refittedForPixelDQM'
)

# Tracking Monitor ####
# Clone for Sim Data
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
TrackMonSim = TrackMon.clone(
    FolderName = 'Tracking/TrackParameters'
)

# Clone for Real Data (Cosmic)
from DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi import *
TrackMonReal = TrackerCosmicTrackMon.clone(
    TrackProducer = 'ctfWithMaterialTracksP5',
    FolderName = 'Tracking/TrackParameters',
    AlgoName = 'CKFTk',
    doSeedParameterHistos = True
)

# Clone for Real Data (Collison)
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
TrackMonColl = TrackMon.clone(
    TrackProducer = 'generalTracks',
    FolderName = 'Tracking/TrackParameters',
    AlgoName = 'CKFTk',
    doSeedParameterHistos = True
)

# Sequences
#removed modules using TkDetMap service
#SiStripSourcesSimData = cms.Sequence(SiStripMonitorTrackSim*MonitorTrackResidualsSim*TrackMonSim)
#SiStripSourcesRealData = cms.Sequence(SiStripMonitorTrackReal*MonitorTrackResidualsReal*TrackMonReal)
#SiStripSourcesRealDataCollision = cms.Sequence(SiStripMonitorTrackColl*MonitorTrackResidualsColl*TrackMonColl)
SiStripSourcesSimData = cms.Sequence(SiStripMonitorDigi*SiStripMonitorCluster*SiStripMonitorTrackSim*MonitorTrackResidualsSim*TrackMonSim)
SiStripSourcesRealData = cms.Sequence(SiStripMonitorDigi*SiStripMonitorCluster*SiStripMonitorTrackReal*MonitorTrackResidualsReal*TrackMonReal)
SiStripSourcesRealDataCollision = cms.Sequence(SiStripMonitorDigi*SiStripMonitorCluster*refittedForPixelDQM*SiStripMonitorTrackColl*MonitorTrackResidualsColl*TrackMonColl)




