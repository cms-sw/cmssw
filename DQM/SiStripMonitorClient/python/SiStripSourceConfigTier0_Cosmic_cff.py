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
SiStripMonitorDigi.TkHistoMapNStripApvShots_On= True
SiStripMonitorDigi.TkHistoMapMedianChargeApvShots_On= False

SiStripMonitorDigi.TH1NApvShots.subdetswitchon = True
SiStripMonitorDigi.TH1NApvShots.globalswitchon = True

SiStripMonitorDigi.TH1ChargeMedianApvShots.subdetswitchon = True
SiStripMonitorDigi.TH1ChargeMedianApvShots.globalswitchon = True

SiStripMonitorDigi.TH1NStripsApvShots.subdetswitchon = True
SiStripMonitorDigi.TH1NStripsApvShots.globalswitchon = True

SiStripMonitorDigi.TH1ApvNumApvShots.subdetswitchon = True
SiStripMonitorDigi.TH1ApvNumApvShots.globalswitchon = True

SiStripMonitorDigi.TProfNShotsVsTime.subdetswitchon = True
SiStripMonitorDigi.TProfNShotsVsTime.globalswitchon = True

SiStripMonitorDigi.TProfGlobalNShots.globalswitchon = True

# SiStripMonitorCluster ####
from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *
SiStripMonitorCluster.Mod_On = False
SiStripMonitorCluster.TProfClustersApvCycle.subdetswitchon = True
SiStripMonitorCluster.TProfTotalNumberOfClusters.subdetswitchon = True 
SiStripMonitorCluster.TrendVsLS = True
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
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_cosmicTk  = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
SiStripMonitorTrack_cosmicTk.Mod_On        = False

# Clone for CKF Tracks
SiStripMonitorTrack_ckf = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_ckf.TrackProducer      = 'ctfWithMaterialTracksP5'
SiStripMonitorTrack_ckf.Mod_On             = False
SiStripMonitorTrack_ckf.TH1nClustersOff.xmax = cms.double(1999.5)

# Clone for Road Search  Tracks
#import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
#SiStripMonitorTrack_rs = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
#SiStripMonitorTrack_rs.TrackProducer       = 'rsWithMaterialTracksP5'
#SiStripMonitorTrack_rs.Mod_On              = False

# TrackerMonitorTrack ####
# Clone for Cosmic Track Finder
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_cosmicTk = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_cosmicTk.trajectoryInput     = 'cosmictrackfinderP5'
MonitorTrackResiduals_cosmicTk.Tracks              = 'cosmictrackfinderP5'
MonitorTrackResiduals_cosmicTk.Mod_On              = False
# Clone for CKF Tracks
MonitorTrackResiduals_ckf = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_ckf.trajectoryInput          = 'ctfWithMaterialTracksP5'
MonitorTrackResiduals_ckf.Tracks                   = 'ctfWithMaterialTracksP5'
MonitorTrackResiduals_ckf.Mod_On                   = False
# Clone for Road Search  Tracks
#import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
#MonitorTrackResiduals_rs = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
#MonitorTrackResiduals_rs.trajectoryInput           = 'rsWithMaterialTracksP5'
#MonitorTrackResiduals_rs.Tracks                    = 'rsWithMaterialTracksP5'
#MonitorTrackResiduals_rs.Mod_On                    = False

# DQM Services
dqmInfoSiStrip = cms.EDAnalyzer("DQMEventInfo",
     subSystemFolder = cms.untracked.string('SiStrip')
)

# Services needed for TkHistoMap
TkDetMap = cms.Service("TkDetMap")
SiStripDetInfoFileReade = cms.Service("SiStripDetInfoFileReader")

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
SiStripDQMTier0 = cms.Sequence(APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorCluster*SiStripMonitorTrack_ckf*MonitorTrackResiduals_ckf*dqmInfoSiStrip)
