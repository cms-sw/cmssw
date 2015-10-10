import FWCore.ParameterSet.Config as cms

# SiStripMonitorCluster
SiStripMonitorCluster = cms.EDAnalyzer("SiStripMonitorCluster",
    ClusterProducerStrip = cms.InputTag('siStripClusters'),
    ClusterProducerPix = cms.InputTag('siPixelClusters'),
                                     
    ResetMEsEachRun = cms.bool(False),

    StripQualityLabel = cms.string(''),

    SelectAllDetectors = cms.bool(False),
    ShowMechanicalStructureView = cms.bool(True),

    ClusterLabel = cms.string(''),

    TkHistoMap_On = cms.bool(True),
                                     
    TopFolderName = cms.string('SiStrip'),

    BPTXfilter     = cms.PSet(),
    PixelDCSfilter = cms.PSet(),
    StripDCSfilter = cms.PSet(),
                                     
    CreateTrendMEs = cms.bool(False),
    TrendVsLS = cms.bool(False),                                       

    Trending = cms.PSet(
        Nbins = cms.int32(600),
        xmin = cms.double(0.0),
        xmax = cms.double(3600.),
        xaxis = cms.string('Event Time in Seconds')
    ),

    TrendingLS = cms.PSet(             
        Nbins = cms.int32(2400),
        xmin = cms.double(0.0),
        xmax = cms.double(150),
        xaxis = cms.string('Lumisection')
        ),

    TH1ClusterNoise = cms.PSet(
        Nbinx          = cms.int32(20),
        xmin           = cms.double(-0.5),
        xmax           = cms.double(9.5),
        layerswitchon  = cms.bool(False),
        moduleswitchon = cms.bool(True)
    ),

    TH1NrOfClusterizedStrips = cms.PSet(
        Nbinx          = cms.int32(20),
        xmin           = cms.double(-0.5),
        xmax           = cms.double(99.5),
        layerswitchon  = cms.bool(True),
        moduleswitchon = cms.bool(True)
    ),
    TH1ClusterPos = cms.PSet(
        Nbinx          = cms.int32(768),
        xmin           = cms.double(-0.5),
        xmax           = cms.double(767.5),
        layerswitchon  = cms.bool(False),
        moduleswitchon = cms.bool(True)
    ),
    TH1ClusterDigiPos = cms.PSet(
        Nbinx          = cms.int32(768),
        xmin           = cms.double(-0.5),
        xmax           = cms.double(767.5),
        layerswitchon  = cms.bool(False),
        moduleswitchon = cms.bool(False)
    ),                                
    TH1ModuleLocalOccupancy = cms.PSet(
        Nbinx          = cms.int32(51),
        xmin           = cms.double(-0.01),
        xmax           = cms.double(1.01),
        layerswitchon  = cms.bool(True),
        moduleswitchon = cms.bool(True)
    ),
    TH1nClusters = cms.PSet(
        Nbinx          = cms.int32(11),
        xmin           = cms.double(-0.5),
        xmax           = cms.double(10.5),
        layerswitchon  = cms.bool(False),
        moduleswitchon = cms.bool(True)
    ),
    TH1ClusterStoN = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-0.5),
        xmax           = cms.double(299.5),
        layerswitchon  = cms.bool(False),
        moduleswitchon = cms.bool(True)
    ),
    TH1ClusterStoNVsPos = cms.PSet(
        Nbinx          = cms.int32(768),
        xmin           = cms.double(-0.5),
        xmax           = cms.double(767.5),
        Nbiny          = cms.int32(100),
        ymin           = cms.double(-0.5),
        ymax           = cms.double(299.5),
        layerswitchon  = cms.bool(False),
        moduleswitchon = cms.bool(False)
    ),
    TH1ClusterCharge = cms.PSet(
        Nbinx          = cms.int32(200),
        xmin           = cms.double(-0.5),        
        xmax           = cms.double(799.5),
        layerswitchon  = cms.bool(False),
        moduleswitchon = cms.bool(True),
        subdetswitchon = cms.bool(False)
    ),
    TH1ClusterWidth = cms.PSet(
        Nbinx          = cms.int32(20),
        xmin           = cms.double(-0.5),
        xmax           = cms.double(19.5),
        layerswitchon  = cms.bool(False),        
        moduleswitchon = cms.bool(True),
        subdetswitchon = cms.bool(False)
    ),

    TProfNumberOfCluster = cms.PSet(
        Nbinx            = cms.int32(100),
        xmin             = cms.double(-0.5),
        xmax             = cms.double(499.5),
        layerswitchon    = cms.bool(False),        
        moduleswitchon   = cms.bool(False)        
    ),
      
    TProfClusterWidth    = cms.PSet(
        Nbinx            = cms.int32(100),
        xmin             = cms.double(-0.5),
        xmax             = cms.double(499.5),
        layerswitchon    = cms.bool(False),        
        moduleswitchon   = cms.bool(False)        
    ),
                                     
    ClusterConditions = cms.PSet(
        minWidth   = cms.double(0.0),
        On         = cms.bool(True),
        maxStoN    = cms.double(10000.0),
        minStoN    = cms.double(0.0),
        maxWidth   = cms.double(10000.0)
    ),

    TProfTotalNumberOfClusters = cms.PSet(
        subdetswitchon = cms.bool(False)
    ),

    TH1TotalNumberOfClusters = cms.PSet(
        Nbinx          = cms.int32(100),
        xmin           = cms.double(-0.5),
        xmax           = cms.double(14999.5),
        subdetswitchon = cms.bool(False)
    ),
                                       
    TProfClustersApvCycle = cms.PSet(
        Nbins = cms.int32(70),
        xmin = cms.double(-0.5),
        xmax = cms.double(69.5),
        Nbinsy = cms.int32(200),
        ymin = cms.double(0.0),
        ymax = cms.double(0.0),
        subdetswitchon = cms.bool(False)
        ),

    TH2ClustersApvCycle = cms.PSet(
        Nbinsx = cms.int32(70),
        xmin   = cms.double(-0.5),
        xmax   = cms.double(69.5),
        Nbinsy = cms.int32(200),
        ymin = cms.double(0.0),
        yfactor = cms.double(0.2),
        subdetswitchon = cms.bool(False)
    ),
                                     
    TProfClustersVsDBxCycle = cms.PSet(
        Nbins = cms.int32(800),
        xmin = cms.double(0.5),
        xmax = cms.double(800.5),
        ymin = cms.double(0.0),
        ymax = cms.double(0.0),
        subdetswitchon = cms.bool(True)
        ),
                                     
    TProf2ApvCycleVsDBx = cms.PSet(
        Nbinsx = cms.int32(70),
        xmin   = cms.double(-0.5),
        xmax   = cms.double(69.5),
        Nbinsy = cms.int32(800),
        ymin   = cms.double(0.5),
        ymax   = cms.double(800.5),
        zmin   = cms.double(0.0),
        zmax   = cms.double(0.0),
        subdetswitchon = cms.bool(False)
        ),
                                     
    TH2ApvCycleVsDBxGlobal = cms.PSet(
        Nbinsx = cms.int32(70),
        xmin   = cms.double(-0.5),
        xmax   = cms.double(69.5),
        Nbinsy = cms.int32(800),
        ymin   = cms.double(0.5),
        ymax   = cms.double(800.5),
        globalswitchon = cms.bool(True)
        ),

    TH2CStripVsCpixel = cms.PSet(
        Nbinsx = cms.int32(150),
        xmin   = cms.double(-0.5),
        xmax   = cms.double(74999.5),
        Nbinsy = cms.int32(50),
        ymin   = cms.double(-0.5),
        ymax   = cms.double(14999.5),
        globalswitchon = cms.bool(True)
        ),
                                       
    MultiplicityRegions = cms.PSet(
        k0 = cms.double(0.097),  # k from linear fit of the diagonal default 0.13 for 2012 data, 0.097 for 2015
        q0 = cms.double(300),   # +/- variation of y axis intercept default 300
        dk0 = cms.double(40),   #+/- variation of k0 (in %) to contain the diagonal zone defoult 40
        MaxClus = cms.double(26000), #Divide Region 2 and Region 3  default 20000 for 2012 data, 26000 for 2015
        MinPix = cms.double(50)  # minimum number of Pix clusters to flag events with zero Si clusters default 50
        ),
                                       
    TH1MultiplicityRegions = cms.PSet(
        Nbinx          = cms.int32(5),
        xmin           = cms.double(0.5),
        xmax           = cms.double(5.5),
        globalswitchon = cms.bool(False)
        ),                                 

    TH1MainDiagonalPosition= cms.PSet(
        Nbinsx          = cms.int32(100),
        xmin           = cms.double(0.),
        xmax           = cms.double(2.),
        globalswitchon = cms.bool(False)
        ),                            
# Nunmber of Cluster in Pixel
    TH1NClusPx = cms.PSet(
        Nbinsx = cms.int32(200),
        xmax = cms.double(19999.5),                      
        xmin = cms.double(-0.5)
        ),
                                       
# Number of Cluster in Strip
    TH1NClusStrip = cms.PSet(
        Nbinsx = cms.int32(500),
        xmax = cms.double(99999.5),                      
        xmin = cms.double(-0.5)
        ),

    TH1StripNoise2ApvCycle = cms.PSet(
        Nbinsx = cms.int32(70),
        xmin   = cms.double(-0.5),
        xmax   = cms.double(69.5),
        globalswitchon = cms.bool(False)
        )
                                       ,
    TH1StripNoise3ApvCycle = cms.PSet(
        Nbinsx = cms.int32(70),
        xmin   = cms.double(-0.5),
        xmax   = cms.double(69.5),
        globalswitchon = cms.bool(False)
        ),
                                       
    NclusVsCycleTimeProf2D = cms.PSet(
        Nbins = cms.int32(70),
        xmin = cms.double(-0.5),
        xmax = cms.double(69.5),
        Nbinsy = cms.int32(90),
        ymin = cms.double(0.),
        ymax = cms.double(90*262144),
        globalswitchon = cms.bool(True)
        ),

    Mod_On = cms.bool(True),
    ClusterHisto = cms.bool(False),

    HistoryProducer = cms.InputTag("consecutiveHEs"),
    ApvPhaseProducer = cms.InputTag("APVPhases"),
            
    UseDCSFiltering = cms.bool(True),
                                       
    ShowControlView = cms.bool(False),
    ShowReadoutView = cms.bool(False)                               
)
