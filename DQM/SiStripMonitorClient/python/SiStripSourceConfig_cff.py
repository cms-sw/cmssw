import FWCore.ParameterSet.Config as cms

# SiStripMonitorPedestal ###
from DQM.SiStripMonitorPedestals.SiStripMonitorPedestals_cfi import *
import copy
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_RealData_cfi import *
# SiStripMonitorDigi ####
SiStripMonitorDigiReal = copy.deepcopy(SiStripMonitorDigi)
import copy
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_SimData_cfi import *
SiStripMonitorDigiSim = copy.deepcopy(SiStripMonitorDigi)
# SiStripMonitorCluster ####
from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *
# SiStripMonitorQuality ####
from DQM.SiStripMonitorPedestals.SiStripMonitorQuality_cfi import *
# TrackingMonitor ####
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
# TrackerMonitorTrack ####
from DQM.TrackerMonitorTrack.MonitorTrackResiduals_cff import *
SiStripSourcesRealData = cms.Sequence(PedsMon*SiStripMonitorDigiReal*SiStripMonitorCluster*QualityMon*TrackMon*DQMMonitorTrackResiduals)
SiStripSourcesSimData = cms.Sequence(SiStripMonitorDigiSim*SiStripMonitorCluster*QualityMon*TrackMon*DQMMonitorTrackResiduals)
PedsMon.OutputMEsInRootFile = False
# use the following flag to select all detectors (e.g. for mtcc data). by default is false
SiStripMonitorDigiReal.SelectAllDetectors = True
# use the following flag to select all detectors (e.g. for mtcc data). by default is false
SiStripMonitorDigiSim.SelectAllDetectors = True
SiStripMonitorCluster.FillSignalNoiseHistos = False
SiStripMonitorCluster.OutputMEsInRootFile = False
SiStripMonitorCluster.SelectAllDetectors = True
QualityMon.dataLabel = ''

