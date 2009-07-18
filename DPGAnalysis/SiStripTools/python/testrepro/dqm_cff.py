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
SiStripMonitorClusterReal.TH1TotalNumberOfClusters.subdetswitchon = True
SiStripMonitorClusterReal.TH1ClusterDigiPos.moduleswitchon        = True
SiStripMonitorClusterReal.TH2ClustersApvCycle.subdetswitchon      = True
SiStripMonitorClusterReal.TH2ClustersApvCycle.yfactor = 0.0002
# Non default parameters for Digi Monitoring
SiStripMonitorDigi.TH2DigiApvCycle.subdetswitchon = True
SiStripMonitorDigi.TH2DigiApvCycle.yfactor = 0.002

