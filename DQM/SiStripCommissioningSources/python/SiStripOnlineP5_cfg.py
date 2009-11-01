import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripOnline")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")

process.MLlog4cplus = cms.Service("MLlog4cplus")

process.MessageLogger = cms.Service("MessageLogger",
    suppressWarning = cms.untracked.vstring(),
    log4cplus = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    suppressDebug = cms.untracked.vstring(),
    debugModules = cms.untracked.vstring(),
    suppressInfo = cms.untracked.vstring()
)

process.DQMStore = cms.Service("DQMStore")

process.FUShmDQMOutputService = cms.Service("FUShmDQMOutputService",
    initialMessageBufferSize = cms.untracked.int32(1000000),
    compressionLevel = cms.int32(1),
    lumiSectionInterval = cms.untracked.int32(20),
    lumiSectionsPerUpdate = cms.double(1.0),
    useCompression = cms.bool(True)
)

process.SiStripConfigDb = cms.Service("SiStripConfigDb",
    UsingDbCache = cms.untracked.bool(True),
    UsingDb = cms.untracked.bool(True),
    SharedMemory = cms.untracked.string('FEDSM00')
)

process.PedestalsFromConfigDb = cms.ESSource("SiStripPedestalsBuilderFromDb")

process.NoiseFromConfigDb = cms.ESSource("SiStripNoiseBuilderFromDb")

process.FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('UNDEFINED')
)

process.SiStripCondObjBuilderFromDb = cms.Service("SiStripCondObjBuilderFromDb")

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.idealMagneticFieldRecordSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('IdealMagneticFieldRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.UniformMagneticFieldESProducer = cms.ESProducer("UniformMagneticFieldESProducer",
    ZFieldInTesla = cms.double(0.0)
)

process.prefer("UniformMagneticFieldESProducer")
process.StripCPEfromTrackAngleESProducer = cms.ESProducer("StripCPEfromTrackAngleESProducer",
    ComponentName = cms.string('StripCPEfromTrackAngle')
)

process.PixelCPEParmErrorESProducer = cms.ESProducer("PixelCPEParmErrorESProducer",
    UseNewParametrization = cms.bool(True),
    ComponentName = cms.string('PixelCPEfromTrackAngle'),
    UseSigma = cms.bool(True),
    PixelErrorParametrization = cms.string('NOTcmsim'),
    Alpha2Order = cms.bool(True)
)

process.ttrhbwr = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    ComponentName = cms.string('WithTrackAngle'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

process.MeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    UseStripStripQualityDB = cms.bool(False),
    OnDemand = cms.bool(False),
    UseStripAPVFiberQualityDB = cms.bool(False),
    DebugStripModuleQualityDB = cms.untracked.bool(False),
    ComponentName = cms.string(''),
    stripClusterProducer = cms.string('siStripClusters'),
    Regional = cms.bool(False),
    DebugStripAPVFiberQualityDB = cms.untracked.bool(False),
    HitMatcher = cms.string('StandardMatcher'),
    DebugStripStripQualityDB = cms.untracked.bool(False),
    pixelClusterProducer = cms.string('siPixelClusters'),
    stripLazyGetterProducer = cms.string(''),
    UseStripModuleQualityDB = cms.bool(False),
    PixelCPE = cms.string('PixelCPEfromTrackAngle')
)

process.source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('FUShmReader'),
    evtsPerLS = cms.untracked.uint32(200)
)

process.digis = cms.EDProducer("SiStripRawToDigiModule",
    ProductLabel = cms.untracked.string('source'),
    AppendedBytes = cms.untracked.int32(0),
    UseFedKey = cms.untracked.bool(True),
    FedEventDumpFreq = cms.untracked.int32(0),
    FedBufferDumpFreq = cms.untracked.int32(0),
    TriggerFedId = cms.untracked.int32(-1),
    ProductInstance = cms.untracked.string(''),
    CreateDigis = cms.untracked.bool(True)
)

process.trackingRunTypeFilter = cms.EDFilter("SiStripCommissioningRunTypeFilter",
    runTypes = cms.vstring('ApvLatency', 
        'FineDelay'),
    InputModuleLabel = cms.InputTag("digis")
)

process.SiStripGainFakeESSource = cms.ESSource("SiStripGainFakeESSource",
    file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat')
)

process.SiStripGainESProducer = cms.ESProducer("SiStripGainESProducer",
    printDebug = cms.untracked.bool(False),
    NormalizationFactor = cms.double(1.0),
    AutomaticNormalization = cms.bool(False)
)

process.SiStripLAFakeESSource = cms.ESSource("SiStripLAFakeESSource",
    TemperatureError = cms.double(10.0),
    Temperature = cms.double(297.0),
    HoleRHAllParameter = cms.double(0.7),
    ChargeMobility = cms.double(480.0),
    HoleBeta = cms.double(1.213),
    HoleSaturationVelocity = cms.double(8370000.0),
    file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
    AppliedVoltage = cms.double(150.0)
)

process.SiStripRecHitMatcherESProducer = cms.ESProducer("SiStripRecHitMatcherESProducer",
    ComponentName = cms.string('StandardMatcher'),
    NSigmaInside = cms.double(3.0)
)

process.SiStripQualityFakeESSource = cms.ESSource("SiStripQualityFakeESSource")

process.SiPixelFakeLorentzAngleESSource = cms.ESSource("SiPixelFakeLorentzAngleESSource",
    file = cms.FileInPath('CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt')
)

process.SiStripThresholdFakeESSource = cms.ESSource("SiStripThresholdFakeESSource",
    file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
    HighTh = cms.double(5.0),
    LowTh = cms.double(2.0)
)

process.siStripZeroSuppression = cms.EDProducer("SiStripZeroSuppression",
    RawDigiProducersList = cms.VPSet(cms.PSet(
        RawDigiProducer = cms.string('SiStripDigis'),
        RawDigiLabel = cms.string('VirginRaw')
    ), 
        cms.PSet(
            RawDigiProducer = cms.string('SiStripDigis'),
            RawDigiLabel = cms.string('ProcessedRaw')
        ), 
        cms.PSet(
            RawDigiProducer = cms.string('SiStripDigis'),
            RawDigiLabel = cms.string('ScopeMode')
        )),
    FEDalgorithm = cms.uint32(4),
    ZeroSuppressionMode = cms.string('SiStripFedZeroSuppression'),
    CutToAvoidSignal = cms.double(3.0),
    CommonModeNoiseSubtractionMode = cms.string('Median')
)

process.siStripClusters = cms.EDProducer("SiStripClusterizer",
    MaxHolesInCluster = cms.int32(0),
    ChannelThreshold = cms.double(2.0),
    DigiProducersList = cms.VPSet(cms.PSet(
        DigiLabel = cms.string('ZeroSuppressed'),
        DigiProducer = cms.string('SiStripDigis')
    ), 
        cms.PSet(
            DigiLabel = cms.string('VirginRaw'),
            DigiProducer = cms.string('siStripZeroSuppression')
        ), 
        cms.PSet(
            DigiLabel = cms.string('ProcessedRaw'),
            DigiProducer = cms.string('siStripZeroSuppression')
        ), 
        cms.PSet(
            DigiLabel = cms.string('ScopeMode'),
            DigiProducer = cms.string('siStripZeroSuppression')
        )),
    ClusterMode = cms.string('ThreeThresholdClusterizer'),
    SeedThreshold = cms.double(3.0),
    SiStripQualityLabel = cms.string(''),
    ClusterThreshold = cms.double(5.0)
)

process.siStripMatchedRecHits = cms.EDProducer("SiStripRecHitConverter",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    Regional = cms.bool(False),
    stereoRecHits = cms.string('stereoRecHit'),
    Matcher = cms.string('StandardMatcher'),
    matchedRecHits = cms.string('matchedRecHit'),
    LazyGetterProducer = cms.string('SiStripRawToClustersFacility'),
    ClusterProducer = cms.string('siStripClusters'),
    VerbosityLevel = cms.untracked.int32(1),
    rphiRecHits = cms.string('rphiRecHit')
)

process.cosmicseedfinder = cms.EDProducer("CosmicSeedGenerator",
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    originZPosition = cms.double(0.0),
    GeometricStructure = cms.untracked.string('STANDARD'),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    MaxNumberOfCosmicClusters = cms.uint32(300),
    SeedPt = cms.double(1.0),
    HitsForSeeds = cms.untracked.string('pairs'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    ptMin = cms.double(0.9),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    doClusterCheck = cms.bool(True),
    originRadius = cms.double(150.0),
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    originHalfLength = cms.double(90.0)
)

process.cosmictrackfinder = cms.EDProducer("CosmicTrackFinder",
    TrajInEvents = cms.bool(True),
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    pixelRecHits = cms.InputTag("dummy","dummy"),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    MinHits = cms.int32(4),
    Chi2Cut = cms.double(300.0),
    TTRHBuilder = cms.string('WithTrackAngle'),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    debug = cms.untracked.bool(True),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    GeometricStructure = cms.untracked.string('STANDARD'),
    cosmicSeeds = cms.InputTag("cosmicseedfinder")
)

process.siStripFineDelayHit = cms.EDProducer("SiStripFineDelayHit",
    TrajInEvent = cms.bool(True),
    SeedsLabel = cms.InputTag("cosmicseedfinder"),
    MaxClusterDistance = cms.double(2.0),
    TracksLabel = cms.InputTag("cosmictrackfinder"),
    ExplorationWindow = cms.uint32(10),
    MagneticField = cms.bool(False),
    cosmic = cms.bool(True),
    InputModuleLabel = cms.InputTag("digis"),
    DigiLabel = cms.InputTag("siStripZeroSuppression","VirginRaw"),
    ClustersLabel = cms.InputTag("siStripClusters"),
    mode = cms.string('DelayScan'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    NoClustering = cms.bool(True),
    NoTracking = cms.bool(False),
    MaxTrackAngle = cms.double(45.0),
    MinTrackMomentum = cms.double(0.0)
)

process.histosA = cms.EDAnalyzer("SiStripCommissioningSource",
    SummaryInputModuleLabel = cms.string('digis'),
    RootFileName = cms.untracked.string('SiStripCommissioningSource'),
    CommissioningTask = cms.untracked.string('UNDEFINED'),
    InputModuleLabel = cms.string('digis'),
    HistoUpdateFreq = cms.untracked.int32(10)
)

process.histosB = cms.EDAnalyzer("SiStripCommissioningSource",
    SummaryInputModuleLabel = cms.string('digis'),
    CommissioningTask = cms.untracked.string('UNDEFINED'),
    HistoUpdateFreq = cms.untracked.int32(10),
    InputModuleLabel = cms.string('siStripFineDelayHit'),
    RootFileName = cms.untracked.string('SiStripCommissioningSource'),
    SignalToNoiseCut = cms.double(3.0)
)

process.consumer = cms.OutputModule("ShmStreamConsumer",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep FEDRawDataCollection_*_*_*'),
    compression_level = cms.untracked.int32(1),
    use_compression = cms.untracked.bool(True),
    max_event_size = cms.untracked.int32(7000000)
)

process.localReco = cms.Sequence(process.siStripZeroSuppression*process.siStripClusters*process.siStripMatchedRecHits)
process.tracking = cms.Sequence(process.cosmicseedfinder*process.cosmictrackfinder*process.siStripFineDelayHit)
process.withoutTk = cms.Path(process.digis*~process.trackingRunTypeFilter+process.histosA)
process.withTk = cms.Path(process.digis*process.trackingRunTypeFilter+process.localReco*process.tracking*process.histosB)
process.e1 = cms.EndPath(process.consumer)
process.siStripZeroSuppression.RawDigiProducersList = cms.VPSet(cms.PSet(
    RawDigiProducer = cms.string('digis'),
    RawDigiLabel = cms.string('VirginRaw')
), 
    cms.PSet(
        RawDigiProducer = cms.string('digis'),
        RawDigiLabel = cms.string('ProcessedRaw')
    ), 
    cms.PSet(
        RawDigiProducer = cms.string('digis'),
        RawDigiLabel = cms.string('ScopeMode')
    ))
process.siStripClusters.DigiProducersList = cms.VPSet(cms.PSet(
    DigiLabel = cms.string('ZeroSuppressed'),
    DigiProducer = cms.string('digis')
), 
    cms.PSet(
        DigiLabel = cms.string('VirginRaw'),
        DigiProducer = cms.string('siStripZeroSuppression')
    ), 
    cms.PSet(
        DigiLabel = cms.string('ProcessedRaw'),
        DigiProducer = cms.string('siStripZeroSuppression')
    ), 
    cms.PSet(
        DigiLabel = cms.string('ScopeMode'),
        DigiProducer = cms.string('siStripZeroSuppression')
    ))
process.siStripClusters.ChannelThreshold = 2.0
process.siStripClusters.SeedThreshold = 3.0
process.siStripClusters.ClusterThreshold = 5.0

