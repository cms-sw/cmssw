import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorHardware.test.buffer_hack_cfi import *
import DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi
# SiStripMonitorPedestal ###
CondDBMonSim = DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi.PedsMon.clone()
import DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi
CondDBMonReal = DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi.PedsMon.clone()
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
# SiStripMonitorTrack ####
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
# TrackerMonitorTrack ####
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
# TrackingMonitor ####
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
SiStripSourcesRealDataTIF = cms.Sequence(HardwareMonitor*CondDBMonReal*SiStripMonitorDigi*SiStripMonitorClusterReal*QualityMonReal)
SiStripSourcesSimData = cms.Sequence(HardwareMonitor*SiStripMonitorDigi*SiStripMonitorClusterSim*QualityMonSim*TrackMon)
HardwareMonitor.rootFile = ''
HardwareMonitor.buildAllHistograms = False
HardwareMonitor.preSwapOn = False
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
SiStripMonitorTrack.TrackProducer = 'TrackRefitter'
SiStripMonitorTrack.TrackLabel = ''
SiStripMonitorTrack.Cluster_src = 'siStripClusters'
SiStripMonitorTrack.FolderName = 'SiStrip/Tracks'
TrackMon.FolderName = 'SiStrip/Tracks'

