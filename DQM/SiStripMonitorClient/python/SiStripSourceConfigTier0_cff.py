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

# SiStripMonitorCluster ####
from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *
SiStripMonitorCluster.Mod_On = False
SiStripMonitorCluster.TProfClustersApvCycle.subdetswitchon = True

# SiStripMonitorTrack ####
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrack.TrackProducer = 'generalTracks'
SiStripMonitorTrack.Mod_On        = False
SiStripMonitorTrack.FolderName    = 'Tracking/TrackParameters'

# TrackerMonitorTrack ####
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResiduals.trajectoryInput = 'generalTracks'
MonitorTrackResiduals.OutputMEsInRootFile = False
MonitorTrackResiduals.Mod_On = False

# TrackingMonitor ####
# Default Tracking
import DQM.TrackingMonitor.TrackingMonitor_cfi
TrackMon_gentk = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
TrackMon_gentk.FolderName          = 'Tracking/TrackParameters'
TrackMon_gentk.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'

# Pixel Less Tracking
import DQM.TrackingMonitor.TrackingMonitor_cfi
TrackMon_noPixel = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
TrackMon_noPixel.FolderName          = 'Tracking/TrackParameters'
TrackMon_noPixel.BSFolderName        = 'Tracking/TrackParameters/BeamSpotParameters'
TrackMon_noPixel.TrackProducer       = 'ctfPixelLess'
TrackMon_noPixel.SeedProducer        = 'globalPixelLessSeeds'
TrackMon_noPixel.TCProducer          = 'ckfTrackCandidatesPixelLess'
TrackMon_noPixel.AlgoName            = 'PxLessTk'  

# DQM Services
dqmInfoSiStrip = cms.EDFilter("DQMEventInfo",
    subSystemFolder = cms.untracked.string('SiStrip')
)

# Services needed for TkHistoMap
TkDetMap = cms.Service("TkDetMap")
SiStripDetInfoFileReade = cms.Service("SiStripDetInfoFileReader")

# Sequence
SiStripDQMTier0 = cms.Sequence(siStripFEDCheck*siStripFEDMonitor*SiStripMonitorDigi*SiStripMonitorTrack*MonitorTrackResiduals*TrackMon_gentk*TrackMon_noPixel*dqmInfoSiStrip)


