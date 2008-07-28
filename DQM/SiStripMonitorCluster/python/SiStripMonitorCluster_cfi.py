import FWCore.ParameterSet.Config as cms

# SiStripMonitorCluster
SiStripMonitorCluster = cms.EDFilter("SiStripMonitorCluster",
    # by default do not write out any file with histograms
    # can overwrite this in .cfg file with: replace SiStripMonitorCluster.OutputMEsInRootFile = true
    OutputMEsInRootFile = cms.bool(False),
    CreateTrendMEs = cms.bool(False),
    Trending = cms.PSet(
        UpdateMode = cms.int32(1),
        Nbins = cms.int32(10),
        ymax = cms.double(10000.0),
        Steps = cms.int32(10),
        xmax = cms.double(10.0),
        xmin = cms.double(0.0),
        ymin = cms.double(0.0)
    ),
    TH1ClusterNoise = cms.PSet(
        xmin = cms.double(0.0),
        layerswitchon = cms.bool(True),
        Nbinx = cms.int32(20),
        xmax = cms.double(10.0),
        moduleswitchon = cms.bool(True)
    ),
    ResetMEsEachRun = cms.bool(False),
    ClusterConditions = cms.PSet(
        minWidth = cms.double(0.0),
        On = cms.bool(True),
        maxStoN = cms.double(10000.0),
        minStoN = cms.double(0.0),
        maxWidth = cms.double(10000.0)
    ),
    TH1NrOfClusterizedStrips = cms.PSet(
        xmin = cms.double(0.0),
        layerswitchon = cms.bool(True),
        Nbinx = cms.int32(20),
        xmax = cms.double(100.0),
        moduleswitchon = cms.bool(True)
    ),
    TH1ClusterPos = cms.PSet(
        xmin = cms.double(-0.5),
        layerswitchon = cms.bool(False),
        Nbinx = cms.int32(768),
        xmax = cms.double(767.5),
        moduleswitchon = cms.bool(True)
    ),
    OutputFileName = cms.string('SiStripMonitorCluster.root'),
    #
    SelectAllDetectors = cms.bool(False),
    ShowMechanicalStructureView = cms.bool(True),
    TH1ModuleLocalOccupancy = cms.PSet(
        xmin = cms.double(-0.5),
        layerswitchon = cms.bool(True),
        Nbinx = cms.int32(20),
        xmax = cms.double(0.95),
        moduleswitchon = cms.bool(True)
    ),
    TH1nClusters = cms.PSet(
        xmin = cms.double(-0.5),
        layerswitchon = cms.bool(False),
        Nbinx = cms.int32(21),
        xmax = cms.double(10.5),
        moduleswitchon = cms.bool(True)
    ),
    ClusterLabel = cms.string(''),
    TH1ClusterStoN = cms.PSet(
        xmin = cms.double(0.0),
        layerswitchon = cms.bool(False),
        Nbinx = cms.int32(60),
        xmax = cms.double(200.0),
        moduleswitchon = cms.bool(True)
    ),
    StripQualityLabel = cms.string(''),
    ClusterProducer = cms.string('siStripClusters'),
    TH1ClusterCharge = cms.PSet(
        xmin = cms.double(0.0),
        layerswitchon = cms.bool(True),
        Nbinx = cms.int32(100),
        xmax = cms.double(500.0),
        moduleswitchon = cms.bool(True)
    ),
    #select detectors
    detectorson = cms.PSet(
        tidon = cms.bool(True),
        tibon = cms.bool(True),
        tecon = cms.bool(True),
        tobon = cms.bool(True)
    ),
    TH1ClusterWidth = cms.PSet(
        xmin = cms.double(-0.5),
        layerswitchon = cms.bool(True),
        Nbinx = cms.int32(20),
        xmax = cms.double(19.5),
        moduleswitchon = cms.bool(True)
    ),
    ShowControlView = cms.bool(False),
    ShowReadoutView = cms.bool(False)
)


