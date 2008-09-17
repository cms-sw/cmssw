import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorHardware.buffer_hack_cfi import *
HardwareMonitor.rootFile = ''
HardwareMonitor.buildAllHistograms = False

# Condition DB Monitoring ###
from DQM.SiStripMonitorSummary.SiStripMonitorCondData_cfi import *

# SiStripMonitorDigi ####
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi import *
SiStripMonitorDigi.SelectAllDetectors = True

# SiStripMonitorCluster ####
# Clone for Sim Data
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
SiStripMonitorClusterReal = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()
SiStripMonitorClusterReal.OutputMEsInRootFile = False
SiStripMonitorClusterReal.SelectAllDetectors = True
# Clone for Real Data
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
SiStripMonitorClusterSim = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()
SiStripMonitorClusterSim.OutputMEsInRootFile = False
SiStripMonitorClusterSim.SelectAllDetectors = True
SiStripMonitorClusterSim.StripQualityLabel = 'test1'

# SiStripMonitorTrack ####
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrack.TrackProducer = 'generalTracks'
SiStripMonitorTrack.FolderName    = 'SiStrip/Tracks'
SiStripMonitorTrack.Mod_On        = True

# TrackerMonitorTrack ####
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
MonitorTrackResiduals.Tracks              = 'generalTracks'
MonitorTrackResiduals.trajectoryInput     = 'generalTracks'
MonitorTrackResiduals.OutputMEsInRootFile = False
MonitorTrackResiduals.Mod_On              = False

# TrackingMonitor ####
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
TrackMon.TrackProducer = 'generalTracks'
TrackMon.AlgoName      = 'GenTk'
TrackMon.FolderName    = 'SiStrip/Tracks'


