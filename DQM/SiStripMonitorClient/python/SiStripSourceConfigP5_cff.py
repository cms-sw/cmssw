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
SiStripMonitorDigi.TkHistoMapNStripApvShots_On= True
SiStripMonitorDigi.TkHistoMapMedianChargeApvShots_On= True
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

# removing some histograms
SiStripMonitorDigi.TH1ADCsCoolestStrip.moduleswitchon = False
SiStripMonitorDigi.TH1ADCsHottestStrip.moduleswitchon = False
SiStripMonitorDigi.TH1DigiADCs.moduleswitchon = False
SiStripMonitorDigi.TH1StripOccupancy.moduleswitchon = False
SiStripMonitorDigi.TH1NumberOfDigis.moduleswitchon = False

from DQM.SiStripMonitorDigi.SiStripBaselineValidator_cfi import *

# SiStripMonitorCluster ####
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
SiStripMonitorClusterReal = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()
SiStripMonitorClusterReal.SelectAllDetectors = True
SiStripMonitorClusterReal.TProfTotalNumberOfClusters.subdetswitchon = True
SiStripMonitorClusterReal.TProfClustersApvCycle.subdetswitchon = True
SiStripMonitorClusterReal.TH2CStripVsCpixel.globalswitchon=True
SiStripMonitorClusterReal.TH1MultiplicityRegions.globalswitchon=True
SiStripMonitorClusterReal.TH1MainDiagonalPosition.globalswitchon=True
SiStripMonitorClusterReal.TH1StripNoise2ApvCycle.globalswitchon=True
SiStripMonitorClusterReal.TH1StripNoise3ApvCycle.globalswitchon=True
SiStripMonitorClusterReal.ClusterHisto = True

# removing some histograms
SiStripMonitorClusterReal.TH1NrOfClusterizedStrips.moduleswitchon = False
SiStripMonitorClusterReal.TH1ClusterNoise.moduleswitchon = False
SiStripMonitorClusterReal.TH1ClusterStoN.moduleswitchon = False
SiStripMonitorClusterReal.TH1ClusterCharge.moduleswitchon = False
SiStripMonitorClusterReal.TH1ClusterWidth.moduleswitchon = False
SiStripMonitorClusterReal.TH1ModuleLocalOccupancy.moduleswitchon = False
SiStripMonitorClusterReal.TH1nClusters.moduleswitchon = False
SiStripMonitorClusterReal.TH1ClusterPos.moduleswitchon = False

# SiStripMonitorTrack ####
# Clone for Cosmic Track Finder
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_cosmicTk = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
SiStripMonitorTrack_cosmicTk.Mod_On        = False

# Clone for CKF Tracks
SiStripMonitorTrack_ckf = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_ckf.TrackProducer      = 'ctfWithMaterialTracksP5'
SiStripMonitorTrack_ckf.Mod_On             = False

# Clone fir Road Search  Tracks
#SiStripMonitorTrack_rs = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
#SiStripMonitorTrack_rs.TrackProducer       = 'rsWithMaterialTracksP5'
#SiStripMonitorTrack_rs.Mod_On              = True

# Clone for General Tracks (for Collision)
SiStripMonitorTrack_gentk = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_gentk.TrackProducer    = 'generalTracks'
SiStripMonitorTrack_gentk.Mod_On           = False

# Clone for Heavy Ion Tracks (for HI Collisions)
SiStripMonitorTrack_hi = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_hi.TrackProducer    = 'hiGeneralTracks'
SiStripMonitorTrack_hi.Mod_On           = True

# TrackerMonitorTrack ####
# Clone for Cosmic Track Finder
#import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
#MonitorTrackResiduals_cosmicTk = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
#MonitorTrackResiduals_cosmicTk.Tracks              = 'cosmictrackfinderP5'
#MonitorTrackResiduals_cosmicTk.trajectoryInput     = 'cosmictrackfinderP5'
#MonitorTrackResiduals_cosmicTk.Mod_On              = False

# Clone for CKF Tracks
#import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
#MonitorTrackResiduals_ckf = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
#MonitorTrackResiduals_ckf.Tracks                   = 'ctfWithMaterialTracksP5'
#MonitorTrackResiduals_ckf.trajectoryInput          = 'ctfWithMaterialTracksP5'
#MonitorTrackResiduals_ckf.Mod_On                   = False

# Clone for Road Search  Tracks
#import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
#MonitorTrackResiduals_rs = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
#MonitorTrackResiduals_rs.Tracks                    = 'rsWithMaterialTracksP5'
#MonitorTrackResiduals_rs.trajectoryInput           = 'rsWithMaterialTracksP5'
#MonitorTrackResiduals_rs.Mod_On                    = False

# Clone for General Track (for Collision data)
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_gentk = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_gentk.Tracks                 = 'generalTracks'
MonitorTrackResiduals_gentk.trajectoryInput        = 'generalTracks'
MonitorTrackResiduals_gentk.Mod_On                 = False

# Clone for Heavy Ion Tracks (for HI Collisions)
#import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
#MonitorTrackResiduals_hi = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
#MonitorTrackResiduals_hi.Tracks                 = 'hiGeneralTracks'
#MonitorTrackResiduals_hi.trajectoryInput        = 'hiGeneralTracks'
#MonitorTrackResiduals_hi.Mod_On                 = False

# Services needed for TkHistoMap
TkDetMap = cms.Service("TkDetMap")
SiStripDetInfoFileReade = cms.Service("SiStripDetInfoFileReader")
