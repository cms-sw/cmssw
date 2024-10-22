# SiStripMonitorDigi ####
from DQM.SiStripMonitorDigi.SiStripMonitorDigi_cfi import *
SiStripMonitorDigiHVOff = SiStripMonitorDigi.clone(
    SelectAllDetectors = True,
    Mod_On = False,
    TkHistoMap_On = False,
    UseDCSFiltering = False,
    TProfTotalNumberOfDigis = SiStripMonitorDigi.TProfTotalNumberOfDigis.clone(
        subdetswitchon = True
    ),
    TProfDigiApvCycle = SiStripMonitorDigi.TProfDigiApvCycle.clone(
        subdetswitchon = False
    ),
    TH1ADCsCoolestStrip = SiStripMonitorDigi.TH1ADCsCoolestStrip.clone(
        moduleswitchon = False,
        layerswitchon = False
    ),
    TH1ADCsHottestStrip = SiStripMonitorDigi.TH1ADCsHottestStrip.clone(
        moduleswitchon = False,
        layerswitchon = False
    ),
    TH1DigiADCs = SiStripMonitorDigi.TH1DigiADCs.clone(
        moduleswitchon = False,
        layerswitchon = False
    ),
    TH1NumberOfDigis = SiStripMonitorDigi.TH1NumberOfDigis.clone(
        moduleswitchon = False,
        layerswitchon = False
    ),
    TH1StripOccupancy = SiStripMonitorDigi.TH1StripOccupancy.clone(
        moduleswitchon = False,
        layerswitchon = False
    )
)

# SiStripMonitorCluster ####
from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *
SiStripMonitorClusterHVOff = SiStripMonitorCluster.clone(
    SelectAllDetectors = True,
    Mod_On = False,
    TkHistoMap_On = False,
    TProfTotalNumberOfClusters = SiStripMonitorCluster.TProfTotalNumberOfClusters.clone(
        subdetswitchon = True
    ),
    TProfClustersApvCycle = SiStripMonitorCluster.TProfClustersApvCycle.clone(
        subdetswitchon = False
    ),
    UseDCSFiltering = False,
    TH1ClusterNoise = SiStripMonitorCluster.TH1ClusterNoise.clone(
        layerswitchon = False,
        moduleswitchon = False
    ),
    TH1NrOfClusterizedStrips = SiStripMonitorCluster.TH1NrOfClusterizedStrips.clone(
        layerswitchon = False,
        moduleswitchon = False
    ),
    TH1ClusterPos = SiStripMonitorCluster.TH1ClusterPos.clone(
        layerswitchon = False,
        moduleswitchon = False
    ),
    TH1ClusterDigiPos = SiStripMonitorCluster.TH1ClusterDigiPos.clone(
        layerswitchon = False,
        moduleswitchon = False
    ),
    TH1ModuleLocalOccupancy = SiStripMonitorCluster.TH1ModuleLocalOccupancy.clone(
        layerswitchon = False,
        moduleswitchon = False
    ),
    TH1nClusters = SiStripMonitorCluster.TH1nClusters.clone(
        layerswitchon = False,
        moduleswitchon = False
    ),
    TH1ClusterStoN = SiStripMonitorCluster.TH1ClusterStoN.clone(
        layerswitchon = False,
        moduleswitchon = False
    ),
    TH1ClusterStoNVsPos = SiStripMonitorCluster.TH1ClusterStoNVsPos.clone(
        layerswitchon = False,
        moduleswitchon = False
    ),
    TH1ClusterCharge = SiStripMonitorCluster.TH1ClusterCharge.clone(
        layerswitchon = False,
        moduleswitchon = False
    ),
    TH1ClusterWidth = SiStripMonitorCluster.TH1ClusterWidth.clone(
        layerswitchon = False,
        moduleswitchon = False
    ),
    TProfClustersVsDBxCycle = SiStripMonitorCluster.TProfClustersVsDBxCycle.clone(
        subdetswitchon = False
    ),
    TH2ApvCycleVsDBxGlobal = SiStripMonitorCluster.TH2ApvCycleVsDBxGlobal.clone(
        globalswitchon = False
    ),
    TH2CStripVsCpixel = SiStripMonitorCluster.TH2CStripVsCpixel.clone(
        globalswitchon = True
    ),
    TH1MultiplicityRegions = SiStripMonitorCluster.TH1MultiplicityRegions.clone(
        globalswitchon = True
    ),
    TH1MainDiagonalPosition = SiStripMonitorCluster.TH1MainDiagonalPosition.clone(
        globalswitchon = True
    ),
    TH1StripNoise2ApvCycle = SiStripMonitorCluster.TH1StripNoise2ApvCycle.clone(
        globalswitchon = True
    ),
    TH1StripNoise3ApvCycle = SiStripMonitorCluster.TH1StripNoise3ApvCycle.clone(
        globalswitchon = True
    )
)
