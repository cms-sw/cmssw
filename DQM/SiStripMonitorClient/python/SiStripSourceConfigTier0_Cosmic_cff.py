import FWCore.ParameterSet.Config as cms

# FED integrity Check
from DQM.SiStripMonitorHardware.siStripFEDCheck_cfi import *
siStripFEDCheck.HistogramUpdateFrequency = 0
siStripFEDCheck.DoPayloadChecks          = True
siStripFEDCheck.CheckChannelLengths      = True
siStripFEDCheck.CheckChannelPacketCodes  = True
siStripFEDCheck.CheckFELengths           = True
siStripFEDCheck.CheckChannelStatus       = True

# FED Monitoring
from DQM.SiStripMonitorHardware.siStripFEDMonitor_Tier0_cff import *

# SiStripMonitorDigi ####
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi import *
SiStripMonitorDigi.Mod_On = False
SiStripMonitorDigi.TProfDigiApvCycle.subdetswitchon = True

# APV shots monitoring
SiStripMonitorDigi.TkHistoMapNApvShots_On = True 
SiStripMonitorDigi.TkHistoMapNStripApvShots_On= False
SiStripMonitorDigi.TkHistoMapMedianChargeApvShots_On= False

SiStripMonitorDigi.TH1NApvShots.subdetswitchon = True
SiStripMonitorDigi.TH1NApvShots.globalswitchon = True

SiStripMonitorDigi.TH1ChargeMedianApvShots.subdetswitchon = False
SiStripMonitorDigi.TH1ChargeMedianApvShots.globalswitchon = True

SiStripMonitorDigi.TH1NStripsApvShots.subdetswitchon = False
SiStripMonitorDigi.TH1NStripsApvShots.globalswitchon = False

SiStripMonitorDigi.TH1ApvNumApvShots.subdetswitchon = False
SiStripMonitorDigi.TH1ApvNumApvShots.globalswitchon = False

SiStripMonitorDigi.TProfNShotsVsTime.subdetswitchon = False
SiStripMonitorDigi.TProfNShotsVsTime.globalswitchon = False

SiStripMonitorDigi.TProfGlobalNShots.globalswitchon = True

# SiStripMonitorCluster ####
from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *
SiStripMonitorCluster.Mod_On = False
SiStripMonitorCluster.TProfClustersApvCycle.subdetswitchon = True
SiStripMonitorCluster.TProfTotalNumberOfClusters.subdetswitchon = True 
SiStripMonitorCluster.TrendVs10LS = False
SiStripMonitorCluster.TH2CStripVsCpixel.globalswitchon=False
SiStripMonitorCluster.TH1MultiplicityRegions.globalswitchon=False
SiStripMonitorCluster.TH1MainDiagonalPosition.globalswitchon=False
SiStripMonitorCluster.TH1StripNoise2ApvCycle.globalswitchon=True
SiStripMonitorCluster.TH1StripNoise3ApvCycle.globalswitchon=True
SiStripMonitorCluster.TH1TotalNumberOfClusters.subdetswitchon = True
SiStripMonitorCluster.TH1TotalNumberOfClusters.xmax = cms.double(1999.5)
SiStripMonitorCluster.ClusterHisto = True
SiStripMonitorCluster.TH1NClusStrip.Nbinsx = cms.int32(100)
SiStripMonitorCluster.TH1NClusStrip.xmax = cms.double(1999.5)
SiStripMonitorCluster.TH1NClusPx.Nbinsx = cms.int32(100)
SiStripMonitorCluster.TH1NClusPx.xmax = cms.double(999.5)

# SiStripMonitorTrack ####
# Clone for Cosmic Tracks
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrack_cosmicTk  = SiStripMonitorTrack.clone(
    TrackProducer = 'cosmictrackfinderP5',
    Mod_On = False
)

# Clone for CKF Tracks
SiStripMonitorTrack_ckf = SiStripMonitorTrack.clone(
    TrackProducer = 'ctfWithMaterialTracksP5',
    Mod_On = False,
    TH1nClustersOff = SiStripMonitorTrack.TH1nClustersOff.clone(
        xmax = 1999.5
    )
)

# Clone for Road Search  Tracks
# SiStripMonitorTrack_rs = SiStripMonitorTrack.clone(
#     TrackProducer = 'rsWithMaterialTracksP5',
#     Mod_On = False
# )

# track refitter 
from RecoTracker.TrackProducer.TrackRefitterP5_cfi import *
refitterForCosmictrackfinderP5 = TrackRefitterP5.clone(
    src = "cosmictrackfinderP5"
)
refitterForCtfWithMaterialTracksP5 = TrackRefitterP5.clone(
    src = "ctfWithMaterialTracksP5"
)
refitterForRsWithMaterialTracksP5 = TrackRefitterP5.clone(
    src = "rsWithMaterialTracksP5"
)

# TrackerMonitorTrack ####
# Clone for Cosmic Track Finder
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResiduals_cosmicTk = MonitorTrackResiduals.clone(
    trajectoryInput = 'refitterForCosmictrackfinderP5',
    Tracks = 'refitterForCosmictrackfinderP5',
    Mod_On = False,
    VertexCut = False
)
# Clone for CKF Tracks
MonitorTrackResiduals_ckf = MonitorTrackResiduals.clone(
    trajectoryInput = 'refitterForCtfWithMaterialTracksP5',
    Tracks = 'refitterForCtfWithMaterialTracksP5',
    Mod_On = False,
    VertexCut = False
)

# Clone for Road Search Tracks
# MonitorTrackResiduals_rs = MonitorTrackResiduals.clone(
#     trajectoryInput = 'refitterForRsWithMaterialTracksP5',
#     Tracks = 'refitterForRsWithMaterialTracksP5',
#     Mod_On = False,
#     VertexCut = False
# )

# DQM Services
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoSiStrip = DQMEDAnalyzer('DQMEventInfo',
     subSystemFolder = cms.untracked.string('SiStrip')
)

# Services needed for TkHistoMap
from CalibTracker.SiStripCommon.TkDetMapESProducer_cfi import *

# Event History Producer
from DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi import *

# APV Phase Producer
from DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi import *

# Sequences 
#SiStripDQMTier0_cosmicTk = cms.Sequence(APVPhases*consecutiveHEs*SiStripMonitorTrack_cosmicTk*MonitorTrackResiduals_cosmicTk*TrackMon_cosmicTk*TrackEffMon_cosmicTk)
SiStripDQMTier0_cosmicTk = cms.Sequence(APVPhases*consecutiveHEs*SiStripMonitorTrack_cosmicTk*MonitorTrackResiduals_cosmicTk)

#removed modules using TkDetMap
#SiStripDQMTier0_ckf = cms.Sequence(APVPhases*consecutiveHEs*MonitorTrackResiduals_ckf*TrackMon_ckf*TrackEffMon_ckf)
SiStripDQMTier0_ckf = cms.Sequence(APVPhases*consecutiveHEs*SiStripMonitorTrack_ckf*MonitorTrackResiduals_ckf)

#SiStripDQMTier0_rs = cms.Sequence(APVPhases*consecutiveHEs*SiStripMonitorTrack_rs*MonitorTrackResiduals_rs*TrackMon_rs*TrackEffMon_rs)

#removed modules using TkDetMap
#SiStripDQMTier0 = cms.Sequence(APVPhases*consecutiveHEs*siStripFEDCheck*MonitorTrackResiduals_ckf*TrackMon_cosmicTk*TrackMon_ckf*TrackEffMon_ckf*TrackSplitMonitor*dqmInfoSiStrip)
#SiStripDQMTier0 = cms.Sequence(APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorCluster*SiStripMonitorTrack_ckf*MonitorTrackResiduals_ckf*TrackMon_cosmicTk*TrackMon_ckf*TrackEffMon_ckf*TrackSplitMonitor*dqmInfoSiStrip)
SiStripDQMTier0 = cms.Sequence(APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorCluster*SiStripMonitorTrack_ckf*refitterForCtfWithMaterialTracksP5*MonitorTrackResiduals_ckf*dqmInfoSiStrip)
