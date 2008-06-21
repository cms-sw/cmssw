import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorHardware.test.buffer_hack_cfi import *
import DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi
# SiStripMonitorPedestal ###
CondDBMonSim = DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi.PedsMon.clone()
import DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi
CondDBMonReal = DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi.PedsMon.clone()
import DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi
PedsMonReal = DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi.PedsMon.clone()
# SiStripMonitorDigi #####
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
SiStripMonitorTrackSim = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
import DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi
SiStripMonitorTrackReal = DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi.SiStripMonitorTrack.clone()
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
# TrackerMonitorTrack ####
MonitorTrackResidualsSim = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
import DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi
MonitorTrackResidualsReal = DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi.MonitorTrackResiduals.clone()
import DQM.TrackingMonitor.TrackingMonitor_cfi
# TrackingMonitor ####
TrackMonSim = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
import DQM.TrackingMonitor.TrackingMonitor_cfi
TrackMonReal = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone()
SiStripSourcesRealDataTIF = cms.Sequence(HardwareMonitor*CondDBMonReal*SiStripMonitorDigi*SiStripMonitorClusterReal*SiStripMonitorTrackReal*MonitorTrackResidualsReal*TrackMonReal)
SiStripSourcesRealData = cms.Sequence(HardwareMonitor*PedsMonReal*SiStripMonitorDigi*SiStripMonitorClusterReal*SiStripMonitorTrackReal*MonitorTrackResidualsReal*TrackMonReal)
SiStripSourcesSimData = cms.Sequence(SiStripMonitorDigi*SiStripMonitorClusterSim*QualityMonSim*SiStripMonitorTrackSim*MonitorTrackResidualsSim*TrackMonSim)
HardwareMonitor.rootFile = ''
HardwareMonitor.buildAllHistograms = False
HardwareMonitor.preSwapOn = False
CondDBMonSim.OutputMEsInRootFile = False
CondDBMonSim.StripQualityLabel = ''
CondDBMonSim.RunTypeFlag = 'ConDBPlotsOnly'
CondDBMonReal.OutputMEsInRootFile = False
CondDBMonReal.StripQualityLabel = 'test1'
CondDBMonReal.RunTypeFlag = 'ConDBPlotsOnly'
PedsMonReal.OutputMEsInRootFile = False
PedsMonReal.StripQualityLabel = 'test1'
PedsMonReal.RunTypeFlag = 'CalculatedPlotsOnly'
SiStripMonitorDigi.SelectAllDetectors = True
SiStripMonitorClusterReal.OutputMEsInRootFile = False
SiStripMonitorClusterReal.SelectAllDetectors = True
SiStripMonitorClusterReal.StripQualityLabel = 'test1'
SiStripMonitorClusterSim.OutputMEsInRootFile = False
SiStripMonitorClusterSim.SelectAllDetectors = True
SiStripMonitorClusterSim.StripQualityLabel = ''
QualityMonReal.StripQualityLabel = 'test1'
QualityMonSim.StripQualityLabel = ''
SiStripMonitorTrackSim.TrackProducer = 'TrackRefitter'
SiStripMonitorTrackSim.TrackLabel = ''
SiStripMonitorTrackSim.Cluster_src = 'siStripClusters'
SiStripMonitorTrackSim.FolderName = 'SiStrip/Tracks'
SiStripMonitorTrackReal.TrackProducer = 'ctfWithMaterialTracksP5'
SiStripMonitorTrackReal.TrackLabel = ''
SiStripMonitorTrackReal.Cluster_src = 'siStripClusters'
SiStripMonitorTrackReal.FolderName = 'SiStrip/Tracks'
MonitorTrackResidualsReal.Tracks = 'ctfWithMaterialTracksP5'
MonitorTrackResidualsReal.trajectoryInput = 'ctfWithMaterialTracksP5'
MonitorTrackResidualsReal.OutputMEsInRootFile = False
TrackMonSim.FolderName = 'SiStrip/Tracks'
TrackMonReal.TrackProducer = 'ctfWithMaterialTracksP5'
TrackMonReal.FolderName = 'SiStrip/Tracks'
TrackMonReal.AlgoName = 'CKFTk'
TrackMonReal.TkSizeMax = 25
TrackMonReal.TkSizeBin = 25

