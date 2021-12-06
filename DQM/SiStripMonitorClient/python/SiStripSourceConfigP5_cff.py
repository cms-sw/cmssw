import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorHardware.siStripFEDMonitor_P5_cff import *
siStripFEDMonitor.nFEDErrorsHistogramConfig.NBins = cms.untracked.uint32(441)
siStripFEDMonitor.nFEDErrorsHistogramConfig.Max = cms.untracked.double(440.5)

# SiStripMonitorDigi ####
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi import *
SiStripMonitorDigi.SelectAllDetectors = True
SiStripMonitorDigi.TProfTotalNumberOfDigis.subdetswitchon = True
SiStripMonitorDigi.TProfDigiApvCycle.subdetswitchon = True
SiStripMonitorDigi.TotalNumberOfDigisFailure.subdetswitchon = True

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

# removing some histograms
SiStripMonitorDigi.TH1ADCsCoolestStrip.moduleswitchon = False
SiStripMonitorDigi.TH1ADCsHottestStrip.moduleswitchon = False
SiStripMonitorDigi.TH1DigiADCs.moduleswitchon = False
SiStripMonitorDigi.TH1StripOccupancy.moduleswitchon = False
SiStripMonitorDigi.TH1NumberOfDigis.moduleswitchon = False

from DQM.SiStripMonitorDigi.SiStripBaselineValidator_cfi import *

# SiStripMonitorCluster ####
from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *
SiStripMonitorClusterReal = SiStripMonitorCluster.clone(
    SelectAllDetectors = True,
    TProfTotalNumberOfClusters = SiStripMonitorCluster.TProfTotalNumberOfClusters.clone(
        subdetswitchon = True
    ),
    TProfClustersApvCycle = SiStripMonitorCluster.TProfClustersApvCycle.clone(
        subdetswitchon = True
    ),
    TH2CStripVsCpixel = SiStripMonitorCluster.TH2CStripVsCpixel.clone(
        globalswitchon = True
    ),
    TH1MultiplicityRegions = SiStripMonitorCluster.TH1MultiplicityRegions.clone(
        globalswitchon = True
    ),
    TH1MainDiagonalPosition = SiStripMonitorCluster.TH1MainDiagonalPosition.clone(
        globalswitchon = True
    ),
    TH1StripNoise2ApvCycle = SiStripMonitorCluster.TH1StripNoise2ApvCycle.clone(
        globalswitchon = True
    ),
    TH1StripNoise3ApvCycle = SiStripMonitorCluster.TH1StripNoise3ApvCycle.clone(
        globalswitchon = True
    ),
    ClusterHisto = True,
    # removing some histograms
    TH1NrOfClusterizedStrips = SiStripMonitorCluster.TH1NrOfClusterizedStrips.clone(
        moduleswitchon = False
    ),
    TH1ClusterNoise = SiStripMonitorCluster.TH1ClusterNoise.clone(
        moduleswitchon = False
    ),
    TH1ClusterStoN = SiStripMonitorCluster.TH1ClusterStoN.clone(
        moduleswitchon = False
    ),
    TH1ClusterCharge = SiStripMonitorCluster.TH1ClusterCharge.clone(
        moduleswitchon = False
    ),
    TH1ClusterWidth = SiStripMonitorCluster.TH1ClusterWidth.clone(
        moduleswitchon = False
    ),
    TH1ModuleLocalOccupancy = SiStripMonitorCluster.TH1ModuleLocalOccupancy.clone(
        moduleswitchon = False
    ),
    TH1nClusters = SiStripMonitorCluster.TH1nClusters.clone(
        moduleswitchon = False
    ),
    TH1ClusterPos = SiStripMonitorCluster.TH1ClusterPos.clone(
        moduleswitchon = False
    )
)

# Clone for Cosmic Track Finder
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_cosmicTk = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone(
    TrackProducer = 'cosmictrackfinderP5',
    Mod_On = False,
)

# Clone for CKF Tracks
SiStripMonitorTrack_ckf = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone(
    TrackProducer = 'ctfWithMaterialTracksP5',
    Mod_On = False,
)

# Clone fir Road Search  Tracks
# SiStripMonitorTrack_rs = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone(
#     TrackProducer = 'rsWithMaterialTracksP5',
#     Mod_On = True,
# )

# Clone for General Tracks (for Collision)
SiStripMonitorTrack_gentk = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone(
    TrackProducer = 'generalTracks',
    Mod_On = False
)

# Clone for Heavy Ion Tracks (for HI Collisions)
SiStripMonitorTrack_hi = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone(
    TrackProducer = 'hiGeneralTracks',
    Mod_On = True
)

# TrackerMonitorTrack ####
# Clone for Cosmic Track Finder
# import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# MonitorTrackResiduals_cosmicTk = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone(
#     Tracks = 'cosmictrackfinderP5',
#     trajectoryInput = 'cosmictrackfinderP5',
#     Mod_On = False,
#     VertexCut = False
# )

# Clone for CKF Tracks
# import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# MonitorTrackResiduals_ckf = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone(
#     Tracks = 'ctfWithMaterialTracksP5',
#     trajectoryInput = 'ctfWithMaterialTracksP5',
#     Mod_On = False
#     VertexCut = False
# )

# Clone for Road Search  Tracks
# import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# MonitorTrackResiduals_rs = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone(
#     Tracks = 'rsWithMaterialTracksP5',
#     trajectoryInput = 'rsWithMaterialTracksP5',
#     Mod_On = False,
#     VertexCut = False
# )

# Clone for General Track (for Collision data)
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_gentk = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone(
    Tracks = 'generalTracks',
    trajectoryInput = 'generalTracks',
    Mod_On = False
)

# Clone for Heavy Ion Tracks (for HI Collisions)
# import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# MonitorTrackResiduals_hi = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone(
#     Tracks = 'hiGeneralTracks',
#     trajectoryInput = 'hiGeneralTracks',
#     Mod_On = False
# )

# Services needed for TkHistoMap
from CalibTracker.SiStripCommon.TkDetMapESProducer_cfi import *
