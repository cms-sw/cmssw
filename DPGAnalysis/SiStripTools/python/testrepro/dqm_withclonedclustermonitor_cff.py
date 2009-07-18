import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# DQM Modules
#-------------------------------------------------

from DQMServices.Core.DQMStore_cfg import *
from DQMServices.Components.DQMEnvironment_cfi import *

dqmSaver.convention   = 'Online'
dqmSaver.saveAtJobEnd = True

#-------------------------------------------------
# DQM Modules
#-------------------------------------------------
from DQM.SiStripMonitorClient.SiStripSourceConfigP5_cff import *

# Non default parameters for Cluster Monitoring
SiStripMonitorClusterReal.ClusterProducer                         = 'calZeroBiasClusters'
SiStripMonitorClusterReal.StripQualityLabel                       = 'unbiased'
SiStripMonitorClusterReal.TProfClustersApvCycle.subdetswitchon = False

# Non default parameters for Digi Monitoring
SiStripMonitorDigi.TH2DigiApvCycle.subdetswitchon = True
SiStripMonitorDigi.TH2DigiApvCycle.yfactor = 0.002

#-------------------------------------------------
# cloned SiStripMonitorCluster
#-------------------------------------------------
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi

ClusterDQM = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()

ClusterDQM.OutputMEsInRootFile                     = False 
ClusterDQM.SelectAllDetectors                      = True 
ClusterDQM.StripQualityLabel                       = 'unbiased'
ClusterDQM.ClusterProducer                         = 'calZeroBiasClusters'

ClusterDQM.TH1ClusterDigiPos.moduleswitchon        = True
ClusterDQM.TH1ClusterDigiPos.layerswitchon         = False
ClusterDQM.TH1ClusterPos.moduleswitchon            = False
ClusterDQM.TH1ClusterPos.layerswitchon             = False
ClusterDQM.TH1nClusters.moduleswitchon             = False
ClusterDQM.TH1nClusters.layerswitchon              = False
ClusterDQM.TH1ClusterStoN.moduleswitchon           = False
ClusterDQM.TH1ClusterStoN.layerswitchon            = False
ClusterDQM.TH1ClusterStoNVsPos.moduleswitchon      = False
ClusterDQM.TH1ClusterStoNVsPos.layerswitchon       = False
ClusterDQM.TH1ClusterNoise.moduleswitchon          = False
ClusterDQM.TH1ClusterNoise.layerswitchon           = False
ClusterDQM.TH1NrOfClusterizedStrips.moduleswitchon = False
ClusterDQM.TH1NrOfClusterizedStrips.layerswitchon  = False
ClusterDQM.TH1ModuleLocalOccupancy.moduleswitchon  = False
ClusterDQM.TH1ModuleLocalOccupancy.layerswitchon   = False
ClusterDQM.TH1ClusterCharge.moduleswitchon         = False
ClusterDQM.TH1ClusterCharge.layerswitchon          = False
ClusterDQM.TH1ClusterWidth.moduleswitchon          = False
ClusterDQM.TH1ClusterWidth.layerswitchon           = False
ClusterDQM.TProfNumberOfCluster.moduleswitchon     = False
ClusterDQM.TProfNumberOfCluster.layerswitchon      = False
ClusterDQM.TProfClusterWidth.moduleswitchon        = False
ClusterDQM.TProfClusterWidth.layerswitchon         = False
ClusterDQM.TkHistoMap_On                           = False
ClusterDQM.TH1TotalNumberOfClusters.subdetswitchon = True
ClusterDQM.TProfClustersApvCycle.subdetswitchon    = True
ClusterDQM.TH2ClustersApvCycle.subdetswitchon      = True
ClusterDQM.TH2ClustersApvCycle.yfactor = 0.0002

