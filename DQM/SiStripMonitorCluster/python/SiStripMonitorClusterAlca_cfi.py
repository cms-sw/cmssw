import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi import *

# SiStripMonitorCluster
SiStripCalZeroBiasMonitorCluster = SiStripMonitorCluster.clone(
    ClusterProducerStrip = "calZeroBiasClusters",
    ClusterProducerPix = 'siPixelClusters',
    ResetMEsEachRun = False,
    StripQualityLabel = 'unbiased',
    SelectAllDetectors = True,
    ShowMechanicalStructureView = True,
    ClusterLabel = '',
    TkHistoMap_On = False,
    ClusterChTkHistoMap_On = False,
    TopFolderName = 'AlcaReco/SiStrip',
    BPTXfilter = SiStripMonitorCluster.BPTXfilter.clone(),
    PixelDCSfilter = SiStripMonitorCluster.PixelDCSfilter.clone(),
    StripDCSfilter = SiStripMonitorCluster.StripDCSfilter.clone(),
    CreateTrendMEs = False,
    TrendVs10LS = False,
    TH1ClusterNoise = SiStripMonitorCluster.TH1ClusterNoise.clone(
        Nbinx = 20,
        xmin = -0.5,
        xmax = 9.5,
        layerswitchon  = False,
        moduleswitchon = False
    ),
    TH1NrOfClusterizedStrips = SiStripMonitorCluster.TH1NrOfClusterizedStrips.clone(
        Nbinx = 20,
        xmin = -0.5,
        xmax = 99.5,
        layerswitchon  = False,
        moduleswitchon = False
    ),
    TH1ClusterPos = SiStripMonitorCluster.TH1ClusterPos.clone(
        Nbinx = 768,
        xmin = -0.5,
        xmax = 767.5,
        layerswitchon  = True,
        moduleswitchon = False
    ),
    TH1ClusterDigiPos = SiStripMonitorCluster.TH1ClusterDigiPos.clone(
        Nbinx = 768,
        xmin = -0.5,
        xmax = 767.5,
        layerswitchon  = False,
        moduleswitchon = True
    ),
    TH1ModuleLocalOccupancy = SiStripMonitorCluster.TH1ModuleLocalOccupancy.clone(
        Nbinx = 20,
        xmin = -0.5,
        xmax = 0.95,
        layerswitchon  = False,
        moduleswitchon = False
    ),
    TH1nClusters = SiStripMonitorCluster.TH1nClusters.clone(
        Nbinx = 11,
        xmin = -0.5,
        xmax = 10.5,
        layerswitchon  = False,
        moduleswitchon = False
    ),
    TH1ClusterStoN = SiStripMonitorCluster.TH1ClusterStoN.clone(
        Nbinx = 100,
        xmin = -0.5,
        xmax = 299.5,
        layerswitchon  = False,
        moduleswitchon = False
    ),
    TH1ClusterStoNVsPos = SiStripMonitorCluster.TH1ClusterStoNVsPos.clone(
        Nbinx = 768,
        xmin = -0.5,
        xmax = 767.5,
        Nbiny = 100,
        ymin = -0.5,
        ymax = 299.5,
        layerswitchon  = False,
        moduleswitchon = False
    ),
    TH1ClusterCharge = SiStripMonitorCluster.TH1ClusterCharge.clone(
        Nbinx = 200,
        xmin = -0.5,        
        xmax = 799.5,
        layerswitchon = False,
        moduleswitchon = False,
        subdetswitchon = True
    ),
    TH1ClusterWidth = SiStripMonitorCluster.TH1ClusterWidth.clone(
        Nbinx = 30,
        xmin = -0.5,
        xmax = 29.5,
        layerswitchon  = False,        
        moduleswitchon = False,
        subdetswitchon = True
    ),
    TProfNumberOfCluster = SiStripMonitorCluster.TProfNumberOfCluster.clone(
        Nbinx = 100,
        xmin = -0.5,
        xmax = 499.5,
        layerswitchon = False,        
        moduleswitchon = False        
    ),
    TProfClusterWidth    = SiStripMonitorCluster.TProfClusterWidth.clone(
        Nbinx = 100,
        xmin = -0.5,
        xmax = 499.5,
        layerswitchon = False,        
        moduleswitchon = False        
    ),
    ClusterConditions = SiStripMonitorCluster.ClusterConditions.clone(
        minWidth = 0.0,
        On = True,
        maxStoN = 10000.0,
        minStoN = 0.0,
        maxWidth = 10000.0
    ),
    TProfTotalNumberOfClusters = SiStripMonitorCluster.TProfTotalNumberOfClusters.clone(
        subdetswitchon = True
    ),
    TH1TotalNumberOfClusters = SiStripMonitorCluster.TH1TotalNumberOfClusters.clone(
        Nbinx = 500,
        xmin = -0.5,
        xmax = 19999.5,
        subdetswitchon = True
    ),
    TProfClustersApvCycle = SiStripMonitorCluster.TProfClustersApvCycle.clone(
        Nbins = 70,
        xmin = -0.5,
        xmax = 69.5,
        Nbinsy = 200,
        ymin = 0.0,
        ymax = 0.0,
        subdetswitchon = True
    ),
    TH2ClustersApvCycle = SiStripMonitorCluster.TH2ClustersApvCycle.clone(
        Nbinsx = 70,
        xmin = -0.5,
        xmax = 69.5,
        Nbinsy = 400,
        ymin = 0.0,
        yfactor = 0.01,
        subdetswitchon = True
    ),
    TProfClustersVsDBxCycle = SiStripMonitorCluster.TProfClustersVsDBxCycle.clone(
        Nbins = 800,
        xmin = 0.5,
        xmax = 800.5,
        ymin = 0.0,
        ymax = 0.0,
        subdetswitchon = False
    ),
    TProf2ApvCycleVsDBx = SiStripMonitorCluster.TProf2ApvCycleVsDBx.clone(
        Nbinsx = 70,
        xmin = -0.5,
        xmax = 69.5,
        Nbinsy = 800,
        ymin = 0.5,
        ymax = 800.5,
        zmin = 0.0,
        zmax = 0.0,
        subdetswitchon = False
    ),
    TH2ApvCycleVsDBxGlobal = SiStripMonitorCluster.TH2ApvCycleVsDBxGlobal.clone(
        Nbinsx = 70,
        xmin = -0.5,
        xmax = 69.5,
        Nbinsy = 800,
        ymin = 0.5,
        ymax = 800.5,
        globalswitchon = False
    ),
    TH2CStripVsCpixel = SiStripMonitorCluster.TH2CStripVsCpixel.clone(
        Nbinsx = 150,
        xmin = -0.5,
        xmax = 74999.5,
        Nbinsy = 50,
        ymin = -0.5,
        ymax = 14999.5,
        globalswitchon = False
    ),
    MultiplicityRegions = SiStripMonitorCluster.MultiplicityRegions.clone(
        k0 = 0.13,  # k from linear fit of the diagonal
        q0 = 300,   # +/- variation of y axis intercept
        dk0 = 40,   #+/- variation of k0 (in %) to contain the diagonal zone
        MaxClus = 20000, #Divide Region 2 and Region 3
        MinPix = 50  # minimum number of Pix clusters to flag events with zero Si clusters
    ),
    TH1MultiplicityRegions = SiStripMonitorCluster.TH1MultiplicityRegions.clone(
        Nbinx = 5,
        xmin = 0.5,
        xmax = 5.5,
        globalswitchon = False
    ),        
    TH1MainDiagonalPosition = SiStripMonitorCluster.TH1MainDiagonalPosition.clone(
        Nbinsx = 100,
        xmin = 0.,
        xmax = 2.,
        globalswitchon = False
    ),   
    # Nunmber of Cluster in Pixel
    TH1NClusPx = SiStripMonitorCluster.TH1NClusPx.clone(
        Nbinsx = 200,
        xmax = 19999.5,                      
        xmin = -0.5
    ),
    # Number of Cluster in Strip
    TH1NClusStrip = SiStripMonitorCluster.TH1NClusStrip.clone(
        Nbinsx = 500,
        xmax = 99999.5,                      
        xmin = -0.5
    ),
    TH1StripNoise2ApvCycle = SiStripMonitorCluster.TH1StripNoise2ApvCycle.clone(
        Nbinsx = 70,
        xmin = -0.5,
        xmax = 69.5,
        globalswitchon = False
    ),
    TH1StripNoise3ApvCycle = SiStripMonitorCluster.TH1StripNoise3ApvCycle.clone(
        Nbinsx = 70,
        xmin = -0.5,
        xmax = 69.5,
        globalswitchon = False
    ),
    Mod_On = True,
    ClusterHisto = False,
    HistoryProducer = "consecutiveHEs",
    ApvPhaseProducer = "APVPhases",
    UseDCSFiltering = True,
    ShowControlView = False,
    ShowReadoutView = False
)
