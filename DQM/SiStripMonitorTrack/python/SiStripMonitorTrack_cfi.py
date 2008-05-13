import FWCore.ParameterSet.Config as cms

# MonitorTrackGlobal
SiStripMonitorTrack = cms.EDFilter("SiStripMonitorTrack",
    OutputMEsInRootFile = cms.bool(False),
    Mod_On = cms.bool(True),
    Trending = cms.PSet(
        UpdateMode = cms.int32(1),
        Nbins = cms.int32(10),
        Steps = cms.int32(5)
    ),
    TH1ClusterNoise = cms.PSet(
        xmin = cms.double(0.0),
        Nbinx = cms.int32(20),
        xmax = cms.double(10.0)
    ),
    FolderName = cms.string('Track/GlobalParameters'),
    #-------------------------------------------------
    #    InputTag Track_src = ctfWithMaterialTracks
    #    InputTag TrackInfo=trackinfoCTF:updatedState
    Cluster_src = cms.InputTag("siStripClusters"),
    #     double ElectronPerAdc      = 313.0 
    #     double EquivalentNoiseCharge300um = 2160.
    #     double BadStripProbability = 0.0
    #     uint32 PedestalValue       = 30
    #     double LowThValue          =  2
    #     double HighThValue         =  5
    TH1nTracks = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(10),
        xmax = cms.double(9.5)
    ),
    ClusterConditions = cms.PSet(
        minWidth = cms.double(0.0),
        On = cms.bool(True),
        maxStoN = cms.double(2000.0),
        minStoN = cms.double(10.0),
        maxWidth = cms.double(200.0)
    ),
    TH1ClusterStoNCorr = cms.PSet(
        xmin = cms.double(0.0),
        Nbinx = cms.int32(200),
        xmax = cms.double(200.0)
    ),
    TH1ClusterPos = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(768),
        xmax = cms.double(767.5)
    ),
    MTCCData = cms.bool(False),
    OutputFileName = cms.string('test_monitortrackparameters_rs.root'),
    TH1nClusters = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(30),
        xmax = cms.double(29.5)
    ),
    TProfileClusterPGV = cms.PSet(
        ymax = cms.double(1.2),
        Nbinx = cms.int32(20),
        Nbiny = cms.int32(20),
        xmax = cms.double(10.0),
        xmin = cms.double(-10.0),
        ymin = cms.double(-0.1)
    ),
    TH1ClusterStoN = cms.PSet(
        xmin = cms.double(0.0),
        Nbinx = cms.int32(60),
        xmax = cms.double(200.0)
    ),
    OffHisto_On = cms.bool(True),
    TH1ClusterChargeCorr = cms.PSet(
        xmin = cms.double(0.0),
        Nbinx = cms.int32(200),
        xmax = cms.double(400.0)
    ),
    TH1ClusterCharge = cms.PSet(
        xmin = cms.double(0.0),
        Nbinx = cms.int32(100),
        xmax = cms.double(500.0)
    ),
    AlgoName = cms.string('GenTk'),
    TH1nRecHits = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(16),
        xmax = cms.double(15.5)
    ),
    TrackProducer = cms.string('generalTracks'),
    ModulesToBeExcluded = cms.vuint32(),
    TH1ClusterWidth = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(20),
        xmax = cms.double(19.5)
    ),
    TrackLabel = cms.string('')
)


