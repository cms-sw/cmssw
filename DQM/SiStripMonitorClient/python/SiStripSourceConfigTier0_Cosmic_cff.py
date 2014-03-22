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

# SiStripMonitorCluster ####
from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *
SiStripMonitorCluster.Mod_On = False
SiStripMonitorCluster.TProfClustersApvCycle.subdetswitchon = True
SiStripMonitorCluster.TProfTotalNumberOfClusters.subdetswitchon = True
SiStripMonitorCluster.TH2CStripVsCpixel.globalswitchon=False
SiStripMonitorCluster.TH1MultiplicityRegions.globalswitchon=False
SiStripMonitorCluster.TH1MainDiagonalPosition.globalswitchon=False
SiStripMonitorCluster.TH1StripNoise2ApvCycle.globalswitchon=True
SiStripMonitorCluster.TH1StripNoise3ApvCycle.globalswitchon=True
SiStripMonitorCluster.TH1TotalNumberOfClusters.subdetswitchon = True
SiStripMonitorCluster.ClusterHisto = True
SiStripMonitorCluster.TH1NClusStrip.Nbinsx = cms.int32(200)
SiStripMonitorCluster.TH1NClusStrip.xmax = cms.double(3999.5)
SiStripMonitorCluster.TH1NClusPx.Nbinsx = cms.int32(100)
SiStripMonitorCluster.TH1NClusPx.xmax = cms.double(999.5)
SiStripMonitorCluster.TH1TotalNumberOfClusters.Nbinx = cms.int32(100)
SiStripMonitorCluster.TH1TotalNumberOfClusters.xmax = cms.double(1999.5)

# SiStripMonitorTrack ####
# Clone for Cosmic Tracks
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_cosmicTk  = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
SiStripMonitorTrack_cosmicTk.Mod_On        = False

# Clone for CKF Tracks
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrack_ckf = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
SiStripMonitorTrack_ckf.TrackProducer      = 'ctfWithMaterialTracksP5'
SiStripMonitorTrack_ckf.Mod_On             = False

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
MonitorTrackResiduals_cosmicTk.OutputMEsInRootFile = False
MonitorTrackResiduals_cosmicTk.Mod_On              = False
# Clone for CKF Tracks
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResiduals_ckf = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
MonitorTrackResiduals_ckf.trajectoryInput          = 'ctfWithMaterialTracksP5'
MonitorTrackResiduals_ckf.OutputMEsInRootFile      = False
MonitorTrackResiduals_ckf.Mod_On                   = False
# Clone for Road Search  Tracks
#import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
#MonitorTrackResiduals_rs = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
#MonitorTrackResiduals_rs.trajectoryInput           = 'rsWithMaterialTracksP5'
#MonitorTrackResiduals_rs.OutputMEsInRootFile       = False
#MonitorTrackResiduals_rs.Mod_On                    = False

# TrackingMonitor ####
# Clone for Cosmic Track Finder
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_cosmicTk = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_cosmicTk.TrackProducer                    = 'cosmictrackfinderP5'
TrackMon_cosmicTk.AlgoName                         = 'CosmicTk'
TrackMon_cosmicTk.FolderName                       = 'Tracking/TrackParameters'
TrackMon_cosmicTk.doSeedParameterHistos            = True

# Clone for CKF Tracks
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_ckf = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_ckf.TrackProducer                         = 'ctfWithMaterialTracksP5'
TrackMon_ckf.AlgoName                              = 'CKFTk'
TrackMon_ckf.FolderName                            = 'Tracking/TrackParameters'
TrackMon_ckf.doSeedParameterHistos                 = True

# Clone for Road Search  Tracks
#import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
#TrackMon_rs = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
#TrackMon_rs.TrackProducer                          = 'rsWithMaterialTracksP5'
#TrackMon_rs.AlgoName                               = 'RSTk'
#TrackMon_rs.FolderName                             = 'Tracking/TrackParameters'
#TrackMon_rs.doSeedParameterHistos                  = True

# Clone for Beam Halo Muon Tracks
import DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi
TrackMon_bhmuon = DQM.TrackingMonitor.TrackerCosmicsTrackingMonitor_cfi.TrackerCosmicTrackMon.clone()
TrackMon_bhmuon.TrackProducer                      = 'ctfWithMaterialTracksBeamHaloMuon'
TrackMon_bhmuon.AlgoName                           = 'BHMuonTk'
TrackMon_bhmuon.FolderName                         = 'Tracking/TrackParameters'
TrackMon_bhmuon.doSeedParameterHistos              = True

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
#import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
#TrackEffMon_rs = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
#TrackEffMon_rs.TKTrackCollection                   = 'rsWithMaterialTracksP5'
#TrackEffMon_rs.AlgoName                            = 'RSTk'
#TrackEffMon_rs.FolderName                          = 'Tracking/TrackParameters/TrackEfficiency'

# Clone for Beam Halo  Tracks
import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
TrackEffMon_bhmuon = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
TrackEffMon_bhmuon.TKTrackCollection               = 'ctfWithMaterialTracksBeamHaloMuon'
TrackEffMon_bhmuon.AlgoName                        = 'BHMuonTk'
TrackEffMon_bhmuon.FolderName                      = 'Tracking/TrackParameters/TrackEfficiency'

# Split Tracking
from  DQM.TrackingMonitor.TrackSplittingMonitor_cfi import *
TrackSplitMonitor.FolderName = 'Tracking/TrackParameters/SplitTracks'


# DQM Services
dqmInfoSiStrip = cms.EDAnalyzer("DQMEventInfo",
     subSystemFolder = cms.untracked.string('SiStrip')
)

# Services needed for TkHistoMap
TkDetMap = cms.Service("TkDetMap")
SiStripDetInfoFileReade = cms.Service("SiStripDetInfoFileReader")

# Event History Producer
from DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi import *

# APV Phase Producer (configuration from DB)
from DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi import *

# Sequences 
SiStripDQMTier0_cosmicTk = cms.Sequence(APVPhases*consecutiveHEs*SiStripMonitorTrack_cosmicTk*MonitorTrackResiduals_cosmicTk*TrackMon_cosmicTk*TrackEffMon_cosmicTk)

SiStripDQMTier0_ckf = cms.Sequence(APVPhases*consecutiveHEs*SiStripMonitorTrack_ckf*MonitorTrackResiduals_ckf*TrackMon_ckf*TrackEffMon_ckf)

#SiStripDQMTier0_rs = cms.Sequence(APVPhases*consecutiveHEs*SiStripMonitorTrack_rs*MonitorTrackResiduals_rs*TrackMon_rs*TrackEffMon_rs)

SiStripDQMTier0 = cms.Sequence(APVPhases*consecutiveHEs*siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorCluster*SiStripMonitorTrack_ckf*MonitorTrackResiduals_ckf*TrackMon_cosmicTk*TrackMon_ckf*TrackEffMon_ckf*TrackSplitMonitor*dqmInfoSiStrip)
