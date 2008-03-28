import FWCore.ParameterSet.Config as cms

# MonitorTrackGlobal
SiStripMonitorTrack = cms.EDFilter("SiStripMonitorTrack",
    OutputMEsInRootFile = cms.bool(False),
    Trending = cms.PSet(
        UpdateMode = cms.int32(1),
        Nbins = cms.int32(10),
        Steps = cms.int32(5)
    ),
    TH1ClusterChargeCorr = cms.PSet(
        xmin = cms.double(0.0),
        Nbinx = cms.int32(200),
        xmax = cms.double(400.0)
    ),
    psfileName = cms.string('insert_ps_filename'),
    #        InputTag Track_src = ctfWithMaterialTracks
    #InputTag Track_src = cosmictrackfinder
    #InputTag Track_src = cosmictrackfindert      #temporary solution
    #    InputTag ClusterInfo_src = siStripClusterInfoProducer
    Cluster_src = cms.InputTag("siStripClusters"),
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
    TrackInfo = cms.InputTag("trackinfoCTF","updatedState"),
    TH1ClusterEta = cms.PSet(
        xmin = cms.double(-1.2),
        Nbinx = cms.int32(100),
        xmax = cms.double(1.2)
    ),
    TH1ClusterStoNCorr = cms.PSet(
        xmin = cms.double(0.0),
        Nbinx = cms.int32(200),
        xmax = cms.double(200.0)
    ),
    EquivalentNoiseCharge300um = cms.double(2160.0),
    PedestalValue = cms.uint32(30),
    TH2ClusterEta = cms.PSet(
        ymax = cms.double(1.2),
        Nbinx = cms.int32(100),
        Nbiny = cms.int32(100),
        xmax = cms.double(1.2),
        xmin = cms.double(-1.2),
        ymin = cms.double(-1.2)
    ),
    #-------------------------------------------------
    Track_src = cms.InputTag("ctfWithMaterialTracks"),
    EtaAlgo = cms.int32(0),
    #    string rawdigiProducer = "SiStripDigis"; 
    #    string rawdigiLabel = "VirginRaw";// "ProcessedRaw" 
    #    InputTag rawdigiProducer = SiStripDigis; 
    #    InputTag rawdigiLabel = VirginRaw;// "ProcessedRaw"  
    #    VPSet RawDigiProducersList = {
    #	{  string RawDigiProducer = "SiStripDigis" string RawDigiLabel = "VirginRaw"    }
    #    }
    #
    MTCCData = cms.bool(False),
    OutputFileName = cms.string('test_monitortrackparameters_rs.root'),
    TH1nClusters = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(30),
        xmax = cms.double(29.5)
    ),
    TH1ClusterPos = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(768),
        xmax = cms.double(767.5)
    ),
    TH1ClusterStoN = cms.PSet(
        xmin = cms.double(0.0),
        Nbinx = cms.int32(60),
        xmax = cms.double(200.0)
    ),
    TH1BadStrips = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(2),
        xmax = cms.double(1.5)
    ),
    Modules = cms.PSet(
        Mod_On = cms.bool(True)
    ),
    fileName = cms.string('insert_root_filename'),
    HighThValue = cms.double(5.0),
    TH1ClusterNoise = cms.PSet(
        xmin = cms.double(0.0),
        Nbinx = cms.int32(20),
        xmax = cms.double(10.0)
    ),
    psfiletype = cms.int32(121),
    TProfileClusterPGV = cms.PSet(
        ymax = cms.double(1.2),
        Nbinx = cms.int32(20),
        Nbiny = cms.int32(20),
        xmax = cms.double(10.0),
        xmin = cms.double(-10.0),
        ymin = cms.double(-0.1)
    ),
    #PSet TH1localAngle = { int32 Nbinx =  721  double xmin = -180.5 double xmax =  180.5 }       
    TH1ClusterCharge = cms.PSet(
        xmin = cms.double(0.0),
        Nbinx = cms.int32(100),
        xmax = cms.double(500.0)
    ),
    TH3ClusterGlobalPos = cms.PSet(
        zmax = cms.double(120.0),
        ymax = cms.double(120.0),
        Nbinz = cms.int32(1000),
        Nbinx = cms.int32(200),
        Nbiny = cms.int32(200),
        xmax = cms.double(120.0),
        xmin = cms.double(-120.0),
        ymin = cms.double(-120.0),
        zmin = cms.double(-0.5)
    ),
    AlgoName = cms.string('ctf'),
    BadStripProbability = cms.double(0.0),
    TH1nRecHits = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(16),
        xmax = cms.double(15.5)
    ),
    TrackProducer = cms.string('ctfWithMaterialTracks'),
    LowThValue = cms.double(2.0),
    TH1TriggerBits = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(10),
        xmax = cms.double(9.5)
    ),
    ModulesToBeExcluded = cms.vuint32(),
    NeighStrips = cms.int32(1),
    TH1ClusterWidth = cms.PSet(
        xmin = cms.double(-0.5),
        Nbinx = cms.int32(20),
        xmax = cms.double(19.5)
    ),
    ElectronPerAdc = cms.double(313.0),
    UseCalibDataFromDB = cms.bool(True),
    TrackLabel = cms.string('')
)


