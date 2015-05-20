import FWCore.ParameterSet.Config as cms

import DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi

# SiStripMonitorCluster
SiStripCalZeroBiasMonitorCluster = DQM.SiStripMonitorCluster.SiStripMonitorCluster_cfi.SiStripMonitorCluster.clone()

SiStripCalZeroBiasMonitorCluster.ClusterProducerStrip = cms.InputTag("calZeroBiasClusters")
SiStripCalZeroBiasMonitorCluster.ClusterProducerPix = cms.InputTag('siPixelClusters')

SiStripCalZeroBiasMonitorCluster.ResetMEsEachRun = cms.bool(False)

SiStripCalZeroBiasMonitorCluster.StripQualityLabel = cms.string('unbiased')

SiStripCalZeroBiasMonitorCluster.SelectAllDetectors = cms.bool(True)
SiStripCalZeroBiasMonitorCluster.ShowMechanicalStructureView = cms.bool(True)

SiStripCalZeroBiasMonitorCluster.ClusterLabel = cms.string('')

SiStripCalZeroBiasMonitorCluster.TkHistoMap_On = cms.bool(False)

SiStripCalZeroBiasMonitorCluster.TopFolderName = cms.string('AlcaReco/SiStrip')

SiStripCalZeroBiasMonitorCluster.BPTXfilter     = cms.PSet()
SiStripCalZeroBiasMonitorCluster.PixelDCSfilter = cms.PSet()
SiStripCalZeroBiasMonitorCluster.StripDCSfilter = cms.PSet()

SiStripCalZeroBiasMonitorCluster.CreateTrendMEs = cms.bool(False)
SiStripCalZeroBiasMonitorCluster.TrendVsLS = cms.bool(True)
SiStripCalZeroBiasMonitorCluster.TH1ClusterNoise = cms.PSet(
    Nbinx          = cms.int32(20),
    xmin           = cms.double(-0.5),
    xmax           = cms.double(9.5),
    layerswitchon  = cms.bool(False),
    moduleswitchon = cms.bool(False)
)
SiStripCalZeroBiasMonitorCluster.TH1NrOfClusterizedStrips = cms.PSet(
    Nbinx          = cms.int32(20),
    xmin           = cms.double(-0.5),
    xmax           = cms.double(99.5),
    layerswitchon  = cms.bool(False),
    moduleswitchon = cms.bool(False)
)
SiStripCalZeroBiasMonitorCluster.TH1ClusterPos = cms.PSet(
    Nbinx          = cms.int32(768),
    xmin           = cms.double(-0.5),
    xmax           = cms.double(767.5),
    layerswitchon  = cms.bool(False),
    moduleswitchon = cms.bool(False)
)
SiStripCalZeroBiasMonitorCluster.TH1ClusterDigiPos = cms.PSet(
    Nbinx          = cms.int32(768),
    xmin           = cms.double(-0.5),
    xmax           = cms.double(767.5),
    layerswitchon  = cms.bool(False),
    moduleswitchon = cms.bool(True)
)                                
SiStripCalZeroBiasMonitorCluster.TH1ModuleLocalOccupancy = cms.PSet(
    Nbinx          = cms.int32(20),
    xmin           = cms.double(-0.5),
    xmax           = cms.double(0.95),
    layerswitchon  = cms.bool(False),
    moduleswitchon = cms.bool(False)
)
SiStripCalZeroBiasMonitorCluster.TH1nClusters = cms.PSet(
    Nbinx          = cms.int32(11),
    xmin           = cms.double(-0.5),
    xmax           = cms.double(10.5),
    layerswitchon  = cms.bool(False),
    moduleswitchon = cms.bool(False)
)
SiStripCalZeroBiasMonitorCluster.TH1ClusterStoN = cms.PSet(
    Nbinx          = cms.int32(100),
    xmin           = cms.double(-0.5),
    xmax           = cms.double(299.5),
    layerswitchon  = cms.bool(False),
    moduleswitchon = cms.bool(False)
)
SiStripCalZeroBiasMonitorCluster.TH1ClusterStoNVsPos = cms.PSet(
    Nbinx          = cms.int32(768),
    xmin           = cms.double(-0.5),
    xmax           = cms.double(767.5),
    Nbiny          = cms.int32(100),
    ymin           = cms.double(-0.5),
    ymax           = cms.double(299.5),
    layerswitchon  = cms.bool(False),
    moduleswitchon = cms.bool(False)
)
SiStripCalZeroBiasMonitorCluster.TH1ClusterCharge = cms.PSet(
    Nbinx          = cms.int32(200),
    xmin           = cms.double(-0.5),        
    xmax           = cms.double(799.5),
    layerswitchon  = cms.bool(False),
    moduleswitchon = cms.bool(False),
    subdetswitchon = cms.bool(True)
)
SiStripCalZeroBiasMonitorCluster.TH1ClusterWidth = cms.PSet(
    Nbinx          = cms.int32(30),
    xmin           = cms.double(-0.5),
    xmax           = cms.double(29.5),
    layerswitchon  = cms.bool(False),        
    moduleswitchon = cms.bool(False),
    subdetswitchon = cms.bool(True)
)
SiStripCalZeroBiasMonitorCluster.TProfNumberOfCluster = cms.PSet(
    Nbinx            = cms.int32(100),
    xmin             = cms.double(-0.5),
    xmax             = cms.double(499.5),
    layerswitchon    = cms.bool(False),        
    moduleswitchon   = cms.bool(False)        
)
SiStripCalZeroBiasMonitorCluster.TProfClusterWidth    = cms.PSet(
    Nbinx            = cms.int32(100),
    xmin             = cms.double(-0.5),
    xmax             = cms.double(499.5),
    layerswitchon    = cms.bool(False),        
    moduleswitchon   = cms.bool(False)        
)                           
SiStripCalZeroBiasMonitorCluster.ClusterConditions = cms.PSet(
    minWidth   = cms.double(0.0),
    On         = cms.bool(True),
    maxStoN    = cms.double(10000.0),
    minStoN    = cms.double(0.0),
    maxWidth   = cms.double(10000.0)
)
SiStripCalZeroBiasMonitorCluster.TProfTotalNumberOfClusters = cms.PSet(
    subdetswitchon = cms.bool(True)
)
SiStripCalZeroBiasMonitorCluster.TH1TotalNumberOfClusters = cms.PSet(
    Nbinx          = cms.int32(500),
    xmin           = cms.double(-0.5),
    xmax           = cms.double(19999.5),
    subdetswitchon = cms.bool(True)
)
SiStripCalZeroBiasMonitorCluster.TProfClustersApvCycle = cms.PSet(
    Nbins = cms.int32(70),
    xmin = cms.double(-0.5),
    xmax = cms.double(69.5),
    Nbinsy = cms.int32(200),
    ymin = cms.double(0.0),
    ymax = cms.double(0.0),
    subdetswitchon = cms.bool(True)
    )
SiStripCalZeroBiasMonitorCluster.TH2ClustersApvCycle = cms.PSet(
    Nbinsx = cms.int32(70),
    xmin = cms.double(-0.5),
    xmax = cms.double(69.5),
    Nbinsy = cms.int32(400),
    ymin = cms.double(0.0),
    yfactor = cms.double(0.01),
    subdetswitchon = cms.bool(True)
)
SiStripCalZeroBiasMonitorCluster.TProfClustersVsDBxCycle = cms.PSet(
    Nbins = cms.int32(800),
    xmin = cms.double(0.5),
    xmax = cms.double(800.5),
    ymin = cms.double(0.0),
    ymax = cms.double(0.0),
    subdetswitchon = cms.bool(False)
    )
SiStripCalZeroBiasMonitorCluster.TProf2ApvCycleVsDBx = cms.PSet(
    Nbinsx = cms.int32(70),
    xmin   = cms.double(-0.5),
    xmax   = cms.double(69.5),
    Nbinsy = cms.int32(800),
    ymin   = cms.double(0.5),
    ymax   = cms.double(800.5),
    zmin   = cms.double(0.0),
    zmax   = cms.double(0.0),
    subdetswitchon = cms.bool(False)
    )
SiStripCalZeroBiasMonitorCluster.TH2ApvCycleVsDBxGlobal = cms.PSet(
    Nbinsx = cms.int32(70),
    xmin   = cms.double(-0.5),
    xmax   = cms.double(69.5),
    Nbinsy = cms.int32(800),
    ymin   = cms.double(0.5),
    ymax   = cms.double(800.5),
    globalswitchon = cms.bool(False)
    )
SiStripCalZeroBiasMonitorCluster.TH2CStripVsCpixel = cms.PSet(
    Nbinsx = cms.int32(150),
    xmin   = cms.double(-0.5),
    xmax   = cms.double(74999.5),
    Nbinsy = cms.int32(50),
    ymin   = cms.double(-0.5),
    ymax   = cms.double(14999.5),
    globalswitchon = cms.bool(False)
    )
SiStripCalZeroBiasMonitorCluster.MultiplicityRegions = cms.PSet(
    k0 = cms.double(0.13),  # k from linear fit of the diagonal
    q0 = cms.double(300),   # +/- variation of y axis intercept
    dk0 = cms.double(40),   #+/- variation of k0 (in %) to contain the diagonal zone
    MaxClus = cms.double(20000), #Divide Region 2 and Region 3
    MinPix = cms.double(50)  # minimum number of Pix clusters to flag events with zero Si clusters
    )
SiStripCalZeroBiasMonitorCluster.TH1MultiplicityRegions = cms.PSet(
    Nbinx          = cms.int32(5),
    xmin           = cms.double(0.5),
    xmax           = cms.double(5.5),
    globalswitchon = cms.bool(False)
    )                                 
SiStripCalZeroBiasMonitorCluster.TH1MainDiagonalPosition= cms.PSet(
    Nbinsx          = cms.int32(100),
    xmin           = cms.double(0.),
    xmax           = cms.double(2.),
    globalswitchon = cms.bool(False)
    )                            
# Nunmber of Cluster in Pixel
SiStripCalZeroBiasMonitorCluster.TH1NClusPx = cms.PSet(
    Nbinsx = cms.int32(200),
    xmax = cms.double(19999.5),                      
    xmin = cms.double(-0.5)
    )
# Number of Cluster in Strip
SiStripCalZeroBiasMonitorCluster.TH1NClusStrip = cms.PSet(
    Nbinsx = cms.int32(500),
    xmax = cms.double(99999.5),                      
    xmin = cms.double(-0.5)
    )
SiStripCalZeroBiasMonitorCluster.TH1StripNoise2ApvCycle = cms.PSet(
    Nbinsx = cms.int32(70),
    xmin   = cms.double(-0.5),
    xmax   = cms.double(69.5),
    globalswitchon = cms.bool(False)
    )
SiStripCalZeroBiasMonitorCluster.TH1StripNoise3ApvCycle = cms.PSet(
    Nbinsx = cms.int32(70),
    xmin   = cms.double(-0.5),
    xmax   = cms.double(69.5),
    globalswitchon = cms.bool(False)
    )
SiStripCalZeroBiasMonitorCluster.Mod_On = cms.bool(True)
SiStripCalZeroBiasMonitorCluster.ClusterHisto = cms.bool(False)                                                  

SiStripCalZeroBiasMonitorCluster.HistoryProducer = cms.InputTag("consecutiveHEs")
SiStripCalZeroBiasMonitorCluster.ApvPhaseProducer = cms.InputTag("APVPhases")

SiStripCalZeroBiasMonitorCluster.UseDCSFiltering = cms.bool(True)
                                              
SiStripCalZeroBiasMonitorCluster.ShowControlView = cms.bool(False)
SiStripCalZeroBiasMonitorCluster.ShowReadoutView = cms.bool(False)

