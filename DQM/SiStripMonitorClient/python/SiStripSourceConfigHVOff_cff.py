# SiStripMonitorDigi ####
import DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi
SiStripMonitorDigiHVOff = DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi.SiStripMonitorDigi.clone()
SiStripMonitorDigiHVOff.SelectAllDetectors  = True
SiStripMonitorDigiHVOff.Mod_On              = False
SiStripMonitorDigiHVOff.TkHistoMap_On       = False
SiStripMonitorDigiHVOff.UseDCSFiltering     = False

SiStripMonitorDigiHVOff.TProfTotalNumberOfDigis.subdetswitchon = True
SiStripMonitorDigiHVOff.TProfDigiApvCycle.subdetswitchon   = False

SiStripMonitorDigiHVOff.TH1ADCsCoolestStrip.moduleswitchon = False
SiStripMonitorDigiHVOff.TH1ADCsCoolestStrip.layerswitchon   = False

SiStripMonitorDigiHVOff.TH1ADCsHottestStrip.moduleswitchon = False
SiStripMonitorDigiHVOff.TH1ADCsHottestStrip.layerswitchon  = False

SiStripMonitorDigiHVOff.TH1DigiADCs.moduleswitchon         = False
SiStripMonitorDigiHVOff.TH1DigiADCs.layerswitchon          = False

SiStripMonitorDigiHVOff.TH1NumberOfDigis.moduleswitchon    = False
SiStripMonitorDigiHVOff.TH1NumberOfDigis.layerswitchon     = False

SiStripMonitorDigiHVOff.TH1StripOccupancy.moduleswitchon   = False
SiStripMonitorDigiHVOff.TH1StripOccupancy.layerswitchon    = False


# SiStripMonitorCluster ####
import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi
SiStripMonitorClusterHVOff = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()
SiStripMonitorClusterHVOff.SelectAllDetectors  = True
SiStripMonitorClusterHVOff.Mod_On              = False
SiStripMonitorClusterHVOff.TkHistoMap_On       = False
SiStripMonitorClusterHVOff.TProfTotalNumberOfClusters.subdetswitchon = True
SiStripMonitorClusterHVOff.TProfClustersApvCycle.subdetswitchon = False
SiStripMonitorClusterHVOff.UseDCSFiltering     = False

SiStripMonitorClusterHVOff.TH1ClusterNoise.layerswitchon = False
SiStripMonitorClusterHVOff.TH1ClusterNoise.moduleswitchon = False

SiStripMonitorClusterHVOff.TH1NrOfClusterizedStrips.layerswitchon = False
SiStripMonitorClusterHVOff.TH1NrOfClusterizedStrips.moduleswitchon = False

SiStripMonitorClusterHVOff.TH1ClusterPos.layerswitchon = False
SiStripMonitorClusterHVOff.TH1ClusterPos.moduleswitchon = False

SiStripMonitorClusterHVOff.TH1ClusterDigiPos.layerswitchon = False
SiStripMonitorClusterHVOff.TH1ClusterDigiPos.moduleswitchon = False

SiStripMonitorClusterHVOff.TH1ModuleLocalOccupancy.layerswitchon = False
SiStripMonitorClusterHVOff.TH1ModuleLocalOccupancy.moduleswitchon = False

SiStripMonitorClusterHVOff.TH1nClusters.layerswitchon = False
SiStripMonitorClusterHVOff.TH1nClusters.moduleswitchon = False

SiStripMonitorClusterHVOff.TH1ClusterStoN.layerswitchon = False
SiStripMonitorClusterHVOff.TH1ClusterStoN.moduleswitchon = False

SiStripMonitorClusterHVOff.TH1ClusterStoNVsPos.layerswitchon = False
SiStripMonitorClusterHVOff.TH1ClusterStoNVsPos.moduleswitchon = False

SiStripMonitorClusterHVOff.TH1ClusterCharge.layerswitchon = False
SiStripMonitorClusterHVOff.TH1ClusterCharge.moduleswitchon = False

SiStripMonitorClusterHVOff.TH1ClusterWidth.layerswitchon = False
SiStripMonitorClusterHVOff.TH1ClusterWidth.moduleswitchon = False

SiStripMonitorClusterHVOff.TProfClustersVsDBxCycle.subdetswitchon = False
SiStripMonitorClusterHVOff.TH2ApvCycleVsDBxGlobal.globalswitchon = False

SiStripMonitorClusterHVOff.TH2CStripVsCpixel.globalswitchon=True
SiStripMonitorClusterHVOff.TH1MultiplicityRegions.globalswitchon=True
SiStripMonitorClusterHVOff.TH1MainDiagonalPosition.globalswitchon=True
SiStripMonitorClusterHVOff.TH1StripNoise2ApvCycle.globalswitchon=True
SiStripMonitorClusterHVOff.TH1StripNoise3ApvCycle.globalswitchon=True
