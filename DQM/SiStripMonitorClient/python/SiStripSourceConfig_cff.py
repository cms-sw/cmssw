import FWCore.ParameterSet.Config as cms

import copy
from DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi import *
# SiStripMonitorPedestal ###
CondDBMonSim = copy.deepcopy(PedsMon)
import copy
from DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi import *
CondDBMonReal = copy.deepcopy(PedsMon)
import copy
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_RealData_cfi import *
# SiStripMonitorDigi ####
SiStripMonitorDigiReal = copy.deepcopy(SiStripMonitorDigi)
import copy
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_SimData_cfi import *
SiStripMonitorDigiSim = copy.deepcopy(SiStripMonitorDigi)
# SiStripMonitorCluster ####
from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *
import copy
from DQM.SiStripMonitorPedestals.SiStripMonitorQuality_cfi import *
# SiStripMonitorQuality ####
QualityMonReal = copy.deepcopy(QualityMon)
import copy
from DQM.SiStripMonitorPedestals.SiStripMonitorQuality_cfi import *
QualityMonSim = copy.deepcopy(QualityMon)
# SiStripMonitorTrack ####
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
# TrackerMonitorTrack ####
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cfi import *
# TrackingMonitor ####
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
SiStripSourcesRealDataTIF = cms.Sequence(CondDBMonReal*SiStripMonitorDigiReal*SiStripMonitorCluster*QualityMonReal)
SiStripSourcesSimData = cms.Sequence(SiStripMonitorDigiSim*SiStripMonitorCluster*QualityMonSim*SiStripMonitorTrack*MonitorTrackResiduals*TrackMon)
CondDBMonSim.OutputMEsInRootFile = False
CondDBMonSim.StripQualityLabel = ''
CondDBMonSim.RunTypeFlag = 'ConDBPlotsOnly'
CondDBMonReal.OutputMEsInRootFile = False
CondDBMonReal.StripQualityLabel = 'test1'
CondDBMonReal.RunTypeFlag = 'ConDBPlotsOnly'
# use the following flag to select all detectors (e.g. for mtcc data). by default is false
SiStripMonitorDigiReal.SelectAllDetectors = True
# use the following flag to select all detectors (e.g. for mtcc data). by default is false
SiStripMonitorDigiSim.SelectAllDetectors = True
SiStripMonitorCluster.OutputMEsInRootFile = False
SiStripMonitorCluster.SelectAllDetectors = True
SiStripMonitorTrack.FolderName = 'SiStrip/Tracks'
QualityMonReal.StripQualityLabel = 'test1'
QualityMonSim.StripQualityLabel = ''
SiStripMonitorTrack.TrackProducer = 'TrackRefitter'
SiStripMonitorTrack.TrackLabel = ''
SiStripMonitorTrack.Cluster_src = 'siStripClusters'
TrackMon.FolderName = 'SiStrip/Tracks'

