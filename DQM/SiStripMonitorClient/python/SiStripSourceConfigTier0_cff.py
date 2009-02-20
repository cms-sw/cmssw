import FWCore.ParameterSet.Config as cms

# FED integrity Check
from DQM.SiStripMonitorHardware.siStripFEDCheck_cfi import *
siStripFEDCheck.HistogramUpdateFrequency = 0
siStripFEDCheck.DoPayloadChecks          = True
siStripFEDCheck.CheckChannelLengths      = True
siStripFEDCheck.CheckChannelPacketCodes  = True
siStripFEDCheck.CheckFELengths           = True
siStripFEDCheck.CheckChannelStatus       = True

# SiStripMonitorDigi ####
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi import *
SiStripMonitorDigi.Mod_On = False

# SiStripMonitorTrack ####
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrack.TrackProducer = 'generalTracks'
SiStripMonitorTrack.Mod_On        = False
SiStripMonitorTrack.FolderName    = 'SiStrip/Tracks'

# TrackerMonitorTrack ####
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResiduals.trajectoryInput = 'generalTracks'
MonitorTrackResiduals.OutputMEsInRootFile = False
MonitorTrackResiduals.Mod_On = False

# TrackingMonitor ####
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
TrackMon.TrackProducer = 'generalTracks'
TrackMon.AlgoName = 'CKFTk'
TrackMon.FolderName = 'SiStrip/Tracks'

# Tracking Efficiency
import DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi
TrackEffMon = DQM.TrackingMonitor.TrackEfficiencyMonitor_cfi.TrackEffMon.clone()
TrackEffMon.TKTrackCollection             = 'generalTracks'
TrackEffMon.AlgoName                      = 'CKFTk'
TrackEffMon.FolderName                    = 'SiStrip/Tracks/Efficiencies'

# DQM Services
dqmInfoSiStrip = cms.EDFilter("DQMEventInfo",
    subSystemFolder = cms.untracked.string('SiStrip')
)

# Sequence
SiStripDQMTier0 = cms.Sequence(siStripFEDCheck*SiStripMonitorDigi*SiStripMonitorTrack*MonitorTrackResiduals*TrackMon*TrackEffMon*dqmInfoSiStrip)


