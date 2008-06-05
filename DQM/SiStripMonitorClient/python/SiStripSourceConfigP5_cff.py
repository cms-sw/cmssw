import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorHardware.test.buffer_hack_cfi import *
import DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi
# SiStripMonitorPedestal ###
CondDBMonSim = DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi.PedsMon.clone()
import DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi
CondDBMonReal = DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi.PedsMon.clone()
# SiStripMonitorDigi ####
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi import *
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
# SiStripMonitorCluster ####
SiStripMonitorClusterReal = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
SiStripMonitorClusterSim = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()
import DQM.SiStripMonitorPedestals.SiStripMonitorQuality_cfi
# SiStripMonitorQuality ####
QualityMonReal = DQM.SiStripMonitorPedestals.SiStripMonitorQuality_cfi.QualityMon.clone()
import DQM.SiStripMonitorPedestals.SiStripMonitorQuality_cfi
QualityMonSim = DQM.SiStripMonitorPedestals.SiStripMonitorQuality_cfi.QualityMon.clone()
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
# SiStripMonitorTrack ####
# Cosmic Track Finder
SiStripMonitorTrack_cosmicTk = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
# CKF Tracks
SiStripMonitorTrack_ckf = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
# Road Search  Tracks
SiStripMonitorTrack_rs = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
# Merged Tracks
SiStripMonitorTrack_p5 = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# TrackerMonitorTrack ####
# Cosmic Track Finder
MonitorTrackResiduals_cosmicTk = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# CKF Tracks
MonitorTrackResiduals_ckf = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# Road Search  Tracks
MonitorTrackResiduals_rs = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# Merged Tracks
MonitorTrackResiduals_p5 = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
import DQM.TrackingMonitor.TrackingMonitor_cfi
# TrackingMonitor ####
# Cosmic Track Finder
TrackMon_cosmicTk = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
import DQM.TrackingMonitor.TrackingMonitor_cfi
# CKF Tracks
TrackMon_ckf = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
import DQM.TrackingMonitor.TrackingMonitor_cfi
# Road Search  Tracks
TrackMon_rs = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
HardwareMonitor.rootFile = ''
HardwareMonitor.buildAllHistograms = False
CondDBMonSim.OutputMEsInRootFile = False
CondDBMonSim.StripQualityLabel = ''
CondDBMonSim.RunTypeFlag = 'ConDBPlotsOnly'
CondDBMonReal.OutputMEsInRootFile = False
CondDBMonReal.StripQualityLabel = 'test1'
CondDBMonReal.RunTypeFlag = 'ConDBPlotsOnly'
SiStripMonitorDigi.SelectAllDetectors = True
SiStripMonitorClusterReal.OutputMEsInRootFile = False
SiStripMonitorClusterReal.SelectAllDetectors = True
SiStripMonitorClusterReal.StripQualityLabel = 'test1'
SiStripMonitorClusterSim.OutputMEsInRootFile = False
SiStripMonitorClusterSim.SelectAllDetectors = True
SiStripMonitorClusterSim.StripQualityLabel = ''
QualityMonReal.StripQualityLabel = 'test1'
QualityMonSim.StripQualityLabel = ''
SiStripMonitorTrack_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
SiStripMonitorTrack_cosmicTk.FolderName = 'SiStrip/Tracks'
SiStripMonitorTrack_ckf.TrackProducer = 'ctfWithMaterialTracksP5'
SiStripMonitorTrack_ckf.FolderName = 'SiStrip/Tracks'
SiStripMonitorTrack_rs.TrackProducer = 'rsWithMaterialTracksP5'
SiStripMonitorTrack_rs.FolderName = 'SiStrip/Tracks'
SiStripMonitorTrack_p5.TrackProducer = 'trackCollectionP5'
SiStripMonitorTrack_p5.FolderName = 'SiStrip/Tracks'
MonitorTrackResiduals_cosmicTk.Tracks = 'cosmictrackfinderP5'
MonitorTrackResiduals_cosmicTk.trajectoryInput = 'cosmictrackfinderP5'
MonitorTrackResiduals_cosmicTk.OutputMEsInRootFile = False
MonitorTrackResiduals_ckf.Tracks = 'ctfWithMaterialTracksP5'
MonitorTrackResiduals_ckf.trajectoryInput = 'ctfWithMaterialTracksP5'
MonitorTrackResiduals_ckf.OutputMEsInRootFile = False
MonitorTrackResiduals_rs.Tracks = 'rsWithMaterialTracksP5'
MonitorTrackResiduals_rs.trajectoryInput = 'rsWithMaterialTracksP5'
MonitorTrackResiduals_rs.OutputMEsInRootFile = False
MonitorTrackResiduals_p5.Tracks = 'trackCollectionP5'
MonitorTrackResiduals_p5.trajectoryInput = 'trackCollectionP5'
MonitorTrackResiduals_p5.OutputMEsInRootFile = False
TrackMon_cosmicTk.TrackProducer = 'cosmictrackfinderP5'
TrackMon_cosmicTk.AlgoName = 'CosmicTk'
TrackMon_cosmicTk.FolderName = 'SiStrip/Tracks'
TrackMon_cosmicTk.TkSizeMax = 25
TrackMon_cosmicTk.TkSizeBin = 25
TrackMon_ckf.TrackProducer = 'ctfWithMaterialTracksP5'
TrackMon_ckf.AlgoName = 'CKFTk'
TrackMon_ckf.FolderName = 'SiStrip/Tracks'
TrackMon_ckf.TkSizeMax = 25
TrackMon_ckf.TkSizeBin = 25
TrackMon_rs.TrackProducer = 'rsWithMaterialTracksP5'
TrackMon_rs.AlgoName = 'RSTk'
TrackMon_rs.FolderName = 'SiStrip/Tracks'
TrackMon_rs.TkSizeMax = 25
TrackMon_rs.TkSizeBin = 25

