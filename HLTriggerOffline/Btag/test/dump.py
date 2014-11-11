HLT_BTagCSV07_v1
HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1
HLT_PFMET120_NoiseCleaned_BTagCSV07_v1
HLT_BTagCSV07_v1
HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1
HLT_PFMET120_NoiseCleaned_BTagCSV07_v1

Reading  config.ini

maxEvents =  10000
CMSSWVER =  CMSSW_X_Y_Z
processname =  HLT
jets (for matching) =  hltSelector4CentralJetsL1FastJet
files =  ['root://xrootd.ba.infn.it///store/relval/CMSSW_7_2_0_pre8/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_LS172_V15-v1/00000/8A425442-2A50-E411-A51B-0025905A60B2.root']
btag_modules  cms.VInputTag(cms.InputTag("hltL3CombinedSecondaryVertexBJetTags"), cms.InputTag("hltL3CombinedSecondaryVertexBJetTags"), cms.InputTag("hltL3CombinedSecondaryVertexBJetTags"))
btag_pathes  cms.vstring('HLT_BTagCSV07_v1', 
    'HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1', 
    'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1')
vertex_modules  cms.VInputTag(cms.InputTag("hltFastPrimaryVertex"), cms.InputTag("hltFastPVPixelVertices"), cms.InputTag("hltVerticesL3"), cms.InputTag("hltFastPrimaryVertex"), cms.InputTag("hltFastPVPixelVertices"), 
    cms.InputTag("hltVerticesL3"), cms.InputTag("hltFastPrimaryVertex"), cms.InputTag("hltFastPVPixelVertices"), cms.InputTag("hltVerticesL3"))
vertex_pathes  cms.vstring('HLT_BTagCSV07_v1', 
    'HLT_BTagCSV07_v1', 
    'HLT_BTagCSV07_v1', 
    'HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1', 
    'HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1', 
    'HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1', 
    'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1', 
    'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1', 
    'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1')

import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTBTAG")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('root://xrootd.ba.infn.it///store/relval/CMSSW_7_2_0_pre8/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_LS172_V15-v1/00000/8A425442-2A50-E411-A51B-0025905A60B2.root')
)
process.hltBtagJetsbyRef = cms.EDProducer("JetPartonMatcher",
    jets = cms.InputTag("hltSelector4CentralJetsL1FastJet"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("hltBtagPartons")
)


process.hltBtagJetsbyValAlgo = cms.EDProducer("JetFlavourIdentifier",
    srcByReference = cms.InputTag("hltBtagJetsbyRef"),
    physicsDefinition = cms.bool(False)
)


process.hltBtagPartons = cms.EDProducer("PartonSelector",
    src = cms.InputTag("genParticles"),
    withLeptons = cms.bool(False)
)


process.randomEngineStateProducer = cms.EDProducer("RandomEngineStateProducer")


process.hltBtagTriggerSelection = cms.EDFilter("TriggerResultsFilter",
    hltResults = cms.InputTag("TriggerResults","","HLT"),
    triggerConditions = cms.vstring('HLT_PFMET170*'),
    throw = cms.bool(True)
)


process.EDMtoMEConverter = cms.EDAnalyzer("EDMtoMEConverter",
    Name = cms.untracked.string('EDMtoMEConverter'),
    convertOnEndLumi = cms.untracked.bool(True),
    lumiInputTag = cms.InputTag("MEtoEDMConverter","MEtoEDMConverterLumi"),
    Frequency = cms.untracked.int32(50),
    runInputTag = cms.InputTag("MEtoEDMConverter","MEtoEDMConverterRun"),
    Verbosity = cms.untracked.int32(0),
    convertOnEndRun = cms.untracked.bool(True)
)


process.VertexValidationVertices = cms.EDAnalyzer("HLTVertexPerformanceAnalyzer",
    TriggerResults = cms.InputTag("TriggerResults","","HLT"),
    HLTPathNames = cms.vstring('HLT_BTagCSV07_v1', 
        'HLT_BTagCSV07_v1', 
        'HLT_BTagCSV07_v1', 
        'HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1', 
        'HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1', 
        'HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1', 
        'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1', 
        'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1', 
        'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1'),
    Vertex = cms.VInputTag(cms.InputTag("hltFastPrimaryVertex"), cms.InputTag("hltFastPVPixelVertices"), cms.InputTag("hltVerticesL3"), cms.InputTag("hltFastPrimaryVertex"), cms.InputTag("hltFastPVPixelVertices"), 
        cms.InputTag("hltVerticesL3"), cms.InputTag("hltFastPrimaryVertex"), cms.InputTag("hltFastPVPixelVertices"), cms.InputTag("hltVerticesL3"))
)


process.bTagPostValidation = cms.EDAnalyzer("HLTBTagHarvestingAnalyzer",
    minTag = cms.double(0.6),
    HLTPathNames = cms.vstring('HLT_BTagCSV07_v1', 
        'HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1', 
        'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1'),
    mcFlavours = cms.PSet(
        uds = cms.vuint32(1, 2, 3),
        c = cms.vuint32(4),
        b = cms.vuint32(5),
        g = cms.vuint32(21),
        light = cms.vuint32(1, 2, 3, 21)
    ),
    histoName = cms.vstring('hltL3CombinedSecondaryVertexBJetTags', 
        'hltL3CombinedSecondaryVertexBJetTags', 
        'hltL3CombinedSecondaryVertexBJetTags')
)


process.bTagValidation = cms.EDAnalyzer("HLTBTagPerformanceAnalyzer",
    HLTPathNames = cms.vstring('HLT_BTagCSV07_v1', 
        'HLT_PFMHT100_SingleCentralJet60_BTagCSV0p6_v1', 
        'HLT_PFMET120_NoiseCleaned_BTagCSV07_v1'),
    JetTag = cms.VInputTag(cms.InputTag("hltL3CombinedSecondaryVertexBJetTags"), cms.InputTag("hltL3CombinedSecondaryVertexBJetTags"), cms.InputTag("hltL3CombinedSecondaryVertexBJetTags")),
    MinJetPT = cms.double(20),
    TriggerResults = cms.InputTag("TriggerResults","","HLT"),
    mcFlavours = cms.PSet(
        uds = cms.vuint32(1, 2, 3),
        c = cms.vuint32(4),
        b = cms.vuint32(5),
        g = cms.vuint32(21),
        light = cms.vuint32(1, 2, 3, 21)
    ),
    mcPartons = cms.InputTag("hltBtagJetMCTools")
)


process.dqmEnv = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('YourSubsystem'),
    eventRateWindow = cms.untracked.double(0.5),
    eventInfoFolder = cms.untracked.string('EventInfo')
)


process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    dirName = cms.untracked.string('.'),
    referenceHandling = cms.untracked.string('all'),
    producer = cms.untracked.string('DQM'),
    workflow = cms.untracked.string('/CMSSW_X_Y_Z/RelVal/TrigVal'),
    forceRunNumber = cms.untracked.int32(1),
    saveByRun = cms.untracked.int32(-1),
    saveAtJobEnd = cms.untracked.bool(True),
    fileFormat = cms.untracked.string('ROOT'),
    version = cms.untracked.int32(1),
    referenceRequireStatus = cms.untracked.int32(100),
    convention = cms.untracked.string('Offline'),
    filterName = cms.untracked.string(''),
    runIsComplete = cms.untracked.bool(False),
    saveByLumiSection = cms.untracked.int32(-1)
)


process.hltBtagJetMCTools = cms.Sequence(process.hltBtagPartons+process.hltBtagJetsbyRef+process.hltBtagJetsbyValAlgo)


process.DQM_BTag = cms.Path(process.hltBtagTriggerSelection+process.hltBtagJetMCTools+process.VertexValidationVertices+process.bTagValidation+process.bTagPostValidation+process.dqmSaver)


process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0),
    collateHistograms = cms.untracked.bool(False),
    enableMultiThread = cms.untracked.bool(False),
    forceResetOnBeginLumi = cms.untracked.bool(False),
    LSbasedMode = cms.untracked.bool(False),
    verboseQT = cms.untracked.int32(0)
)


process.MessageLogger = cms.Service("MessageLogger",
    suppressInfo = cms.untracked.vstring(),
    debugs = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    suppressDebug = cms.untracked.vstring(),
    cout = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    cerr_stats = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        output = cms.untracked.string('cerr'),
        optionalPSet = cms.untracked.bool(True)
    ),
    warnings = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(

    ),
    statistics = cms.untracked.vstring('cerr_stats'),
    cerr = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('INFO'),
        FwkJob = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        FwkSummary = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000)
        ),
        optionalPSet = cms.untracked.bool(True)
    ),
    FrameworkJobReport = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        optionalPSet = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000)
        )
    ),
    suppressWarning = cms.untracked.vstring(),
    errors = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('warnings', 
        'errors', 
        'infos', 
        'debugs', 
        'cout', 
        'cerr'),
    debugModules = cms.untracked.vstring(),
    infos = cms.untracked.PSet(
        optionalPSet = cms.untracked.bool(True),
        Root_NoDictionary = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(0)
        ),
        placeholder = cms.untracked.bool(True)
    ),
    categories = cms.untracked.vstring('FwkJob', 
        'FwkReport', 
        'FwkSummary', 
        'Root_NoDictionary'),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport')
)


process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    horeco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = cms.untracked.string('TRandom3')
    ),
    paramMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(54525),
        engineName = cms.untracked.string('TRandom3')
    ),
    saveFileName = cms.untracked.string(''),
    hbhereco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = cms.untracked.string('TRandom3')
    ),
    simSiStripDigiSimLink = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    externalLHEProducer = cms.PSet(
        initialSeed = cms.untracked.uint32(234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    famosPileUp = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    simMuonDTDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    siTrackerGaussianSmearingRecHits = cms.PSet(
        initialSeed = cms.untracked.uint32(24680),
        engineName = cms.untracked.string('TRandom3')
    ),
    ecalPreshowerRecHit = cms.PSet(
        initialSeed = cms.untracked.uint32(6541321),
        engineName = cms.untracked.string('TRandom3')
    ),
    generator = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simMuonRPCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(1234567),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    hiSignal = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    mixSimCaloHits = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    mix = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    VtxSmeared = cms.PSet(
        initialSeed = cms.untracked.uint32(98765432),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    LHCTransport = cms.PSet(
        initialSeed = cms.untracked.uint32(87654321),
        engineName = cms.untracked.string('TRandom3')
    ),
    ecalRecHit = cms.PSet(
        initialSeed = cms.untracked.uint32(654321),
        engineName = cms.untracked.string('TRandom3')
    ),
    hfreco = cms.PSet(
        initialSeed = cms.untracked.uint32(541321),
        engineName = cms.untracked.string('TRandom3')
    ),
    mixRecoTracks = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    hiSignalG4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(11),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    famosSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(13579),
        engineName = cms.untracked.string('TRandom3')
    ),
    MuonSimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(987346),
        engineName = cms.untracked.string('TRandom3')
    ),
    g4SimHits = cms.PSet(
        initialSeed = cms.untracked.uint32(11),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    hiSignalLHCTransport = cms.PSet(
        initialSeed = cms.untracked.uint32(88776655),
        engineName = cms.untracked.string('TRandom3')
    ),
    mixGenPU = cms.PSet(
        initialSeed = cms.untracked.uint32(918273),
        engineName = cms.untracked.string('TRandom3')
    ),
    l1ParamMuons = cms.PSet(
        initialSeed = cms.untracked.uint32(6453209),
        engineName = cms.untracked.string('TRandom3')
    ),
    simBeamSpotFilter = cms.PSet(
        initialSeed = cms.untracked.uint32(87654321),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    simMuonCSCDigis = cms.PSet(
        initialSeed = cms.untracked.uint32(11223344),
        engineName = cms.untracked.string('HepJamesRandom')
    ),
    mixData = cms.PSet(
        initialSeed = cms.untracked.uint32(12345),
        engineName = cms.untracked.string('HepJamesRandom')
    )
)


process.CSCGeometryESModule = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string(''),
    useDDD = cms.bool(True),
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(True),
    alignmentsLabel = cms.string(''),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.bool(True)
)


process.CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL', 
        'ZDC', 
        'CASTOR', 
        'EcalBarrel', 
        'EcalEndcap', 
        'EcalPreshower', 
        'TOWER')
)


process.CaloTopologyBuilder = cms.ESProducer("CaloTopologyBuilder")


process.CaloTowerHardcodeGeometryEP = cms.ESProducer("CaloTowerHardcodeGeometryEP")


process.CastorHardcodeGeometryEP = cms.ESProducer("CastorHardcodeGeometryEP")


process.DTGeometryESModule = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string(''),
    fromDDD = cms.bool(True),
    applyAlignment = cms.bool(True),
    alignmentsLabel = cms.string('')
)


process.EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP",
    applyAlignment = cms.bool(False)
)


process.EcalElectronicsMappingBuilder = cms.ESProducer("EcalElectronicsMappingBuilder")


process.EcalEndcapGeometryEP = cms.ESProducer("EcalEndcapGeometryEP",
    applyAlignment = cms.bool(False)
)


process.EcalPreshowerGeometryEP = cms.ESProducer("EcalPreshowerGeometryEP",
    applyAlignment = cms.bool(False)
)


process.EcalTrigTowerConstituentsMapBuilder = cms.ESProducer("EcalTrigTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/EcalMapping/data/EndCap_TTMap.txt')
)


process.GlobalTrackingGeometryESProducer = cms.ESProducer("GlobalTrackingGeometryESProducer")


process.HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP",
    HcalReLabel = cms.PSet(
        RelabelRules = cms.untracked.PSet(
            Eta16 = cms.untracked.vint32(1, 1, 2, 2, 2, 
                2, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            Eta17 = cms.untracked.vint32(1, 1, 2, 2, 3, 
                3, 3, 4, 4, 4, 
                4, 4, 5, 5, 5, 
                5, 5, 5, 5),
            Eta1 = cms.untracked.vint32(1, 2, 2, 2, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3, 3, 
                3, 3, 3, 3),
            CorrectPhi = cms.untracked.bool(False)
        ),
        RelabelHits = cms.untracked.bool(False)
    )
)


process.MuonDetLayerGeometryESProducer = cms.ESProducer("MuonDetLayerGeometryESProducer")


process.MuonNumberingInitialization = cms.ESProducer("MuonNumberingInitialization")


process.RPCGeometryESModule = cms.ESProducer("RPCGeometryESModule",
    useDDD = cms.untracked.bool(True),
    compatibiltyWith11 = cms.untracked.bool(True)
)


process.TrackerRecoGeometryESProducer = cms.ESProducer("TrackerRecoGeometryESProducer")


process.ZdcHardcodeGeometryEP = cms.ESProducer("ZdcHardcodeGeometryEP")


process.fakeForIdealAlignment = cms.ESProducer("FakeAlignmentProducer",
    appendToDataLabel = cms.string('fakeForIdeal')
)


process.hcalTopologyIdeal = cms.ESProducer("HcalTopologyIdealEP",
    Exclude = cms.untracked.string(''),
    appendToDataLabel = cms.string(''),
    hcalTopologyConstants = cms.PSet(
        maxDepthHE = cms.int32(3),
        maxDepthHB = cms.int32(2),
        mode = cms.string('HcalTopologyMode::LHC')
    )
)


process.idealForDigiCSCGeometry = cms.ESProducer("CSCGeometryESModule",
    appendToDataLabel = cms.string('idealForDigi'),
    useDDD = cms.bool(True),
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(True),
    alignmentsLabel = cms.string('fakeForIdeal'),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.bool(False)
)


process.idealForDigiDTGeometry = cms.ESProducer("DTGeometryESModule",
    appendToDataLabel = cms.string('idealForDigi'),
    fromDDD = cms.bool(True),
    applyAlignment = cms.bool(False),
    alignmentsLabel = cms.string('fakeForIdeal')
)


process.idealForDigiTrackerGeometry = cms.ESProducer("TrackerDigiGeometryESModule",
    appendToDataLabel = cms.string('idealForDigi'),
    fromDDD = cms.bool(True),
    trackerGeometryConstants = cms.PSet(
        ROCS_X = cms.int32(0),
        ROCS_Y = cms.int32(0),
        upgradeGeometry = cms.bool(False),
        BIG_PIX_PER_ROC_Y = cms.int32(2),
        BIG_PIX_PER_ROC_X = cms.int32(1),
        ROWS_PER_ROC = cms.int32(80),
        COLS_PER_ROC = cms.int32(52)
    ),
    applyAlignment = cms.bool(False),
    alignmentsLabel = cms.string('fakeForIdeal')
)


process.l1GtBoardMaps = cms.ESProducer("L1GtBoardMapsTrivialProducer",
    CableList = cms.vstring('Free', 
        'Free', 
        'Free', 
        'TechTr', 
        'IsoEGQ', 
        'NoIsoEGQ', 
        'CenJetQ', 
        'ForJetQ', 
        'TauJetQ', 
        'ESumsQ', 
        'HfQ', 
        'Free', 
        'Free', 
        'Free', 
        'Free', 
        'Free', 
        'MQF4', 
        'MQF3', 
        'MQB2', 
        'MQB1', 
        'MQF8', 
        'MQF7', 
        'MQB6', 
        'MQB5', 
        'MQF12', 
        'MQF11', 
        'MQB10', 
        'MQB9'),
    ActiveBoardsDaqRecord = cms.vint32(-1, 0, 1, 2, 3, 
        4, 5, 6, 7, 8, 
        -1, -1),
    CableToPsbMap = cms.vint32(0, 0, 0, 0, 1, 
        1, 1, 1, 2, 2, 
        2, 2, 3, 3, 3, 
        3, 4, 4, 4, 4, 
        5, 5, 5, 5, 6, 
        6, 6, 6),
    BoardPositionDaqRecord = cms.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 
        -1, -1),
    BoardPositionEvmRecord = cms.vint32(1, 3, -1, -1, -1, 
        -1, -1, -1, -1, -1, 
        2, -1),
    BoardList = cms.vstring('GTFE', 
        'FDL', 
        'PSB', 
        'PSB', 
        'PSB', 
        'PSB', 
        'PSB', 
        'PSB', 
        'PSB', 
        'GMT', 
        'TCS', 
        'TIM'),
    PsbInput = cms.VPSet(cms.PSet(
        Slot = cms.int32(9),
        Ch0 = cms.vstring('TechTrig'),
        Ch1 = cms.vstring('TechTrig'),
        Ch2 = cms.vstring(),
        Ch3 = cms.vstring(),
        Ch4 = cms.vstring(),
        Ch5 = cms.vstring(),
        Ch6 = cms.vstring(),
        Ch7 = cms.vstring()
    ), 
        cms.PSet(
            Slot = cms.int32(13),
            Ch0 = cms.vstring('ForJet', 
                'ForJet'),
            Ch1 = cms.vstring('ForJet', 
                'ForJet'),
            Ch2 = cms.vstring('CenJet', 
                'CenJet'),
            Ch3 = cms.vstring('CenJet', 
                'CenJet'),
            Ch4 = cms.vstring('NoIsoEG', 
                'NoIsoEG'),
            Ch5 = cms.vstring('NoIsoEG', 
                'NoIsoEG'),
            Ch6 = cms.vstring('IsoEG', 
                'IsoEG'),
            Ch7 = cms.vstring('IsoEG', 
                'IsoEG')
        ), 
        cms.PSet(
            Slot = cms.int32(14),
            Ch0 = cms.vstring(),
            Ch1 = cms.vstring(),
            Ch2 = cms.vstring('HfBitCounts', 
                'HfRingEtSums'),
            Ch3 = cms.vstring(),
            Ch4 = cms.vstring('ETT', 
                'HTT'),
            Ch5 = cms.vstring('ETM', 
                'ETM'),
            Ch6 = cms.vstring('TauJet', 
                'TauJet'),
            Ch7 = cms.vstring('TauJet', 
                'TauJet')
        ), 
        cms.PSet(
            Slot = cms.int32(15),
            Ch0 = cms.vstring(),
            Ch1 = cms.vstring(),
            Ch2 = cms.vstring(),
            Ch3 = cms.vstring(),
            Ch4 = cms.vstring(),
            Ch5 = cms.vstring(),
            Ch6 = cms.vstring(),
            Ch7 = cms.vstring()
        ), 
        cms.PSet(
            Slot = cms.int32(19),
            Ch0 = cms.vstring(),
            Ch1 = cms.vstring(),
            Ch2 = cms.vstring(),
            Ch3 = cms.vstring(),
            Ch4 = cms.vstring(),
            Ch5 = cms.vstring(),
            Ch6 = cms.vstring(),
            Ch7 = cms.vstring()
        ), 
        cms.PSet(
            Slot = cms.int32(20),
            Ch0 = cms.vstring(),
            Ch1 = cms.vstring(),
            Ch2 = cms.vstring(),
            Ch3 = cms.vstring(),
            Ch4 = cms.vstring(),
            Ch5 = cms.vstring(),
            Ch6 = cms.vstring(),
            Ch7 = cms.vstring()
        ), 
        cms.PSet(
            Slot = cms.int32(21),
            Ch0 = cms.vstring(),
            Ch1 = cms.vstring(),
            Ch2 = cms.vstring(),
            Ch3 = cms.vstring(),
            Ch4 = cms.vstring(),
            Ch5 = cms.vstring(),
            Ch6 = cms.vstring(),
            Ch7 = cms.vstring()
        )),
    BoardHexNameMap = cms.vint32(0, 253, 187, 187, 187, 
        187, 187, 187, 187, 221, 
        204, 173),
    ActiveBoardsEvmRecord = cms.vint32(-1, 1, -1, -1, -1, 
        -1, -1, -1, -1, -1, 
        0, -1),
    BoardSlotMap = cms.vint32(17, 10, 9, 13, 14, 
        15, 19, 20, 21, 18, 
        7, 16),
    BoardIndex = cms.vint32(0, 0, 0, 1, 2, 
        3, 4, 5, 6, 0, 
        0, 0)
)


process.l1GtParameters = cms.ESProducer("L1GtParametersTrivialProducer",
    EvmActiveBoards = cms.uint32(65535),
    DaqNrBxBoard = cms.vint32(3, 3, 3, 3, 3, 
        3, 3, 3, 3),
    DaqActiveBoards = cms.uint32(65535),
    TotalBxInEvent = cms.int32(3),
    EvmNrBxBoard = cms.vint32(1, 3),
    BstLengthBytes = cms.uint32(30)
)


process.l1GtPrescaleFactorsAlgoTrig = cms.ESProducer("L1GtPrescaleFactorsAlgoTrigTrivialProducer",
    PrescaleFactorsSet = cms.VPSet(cms.PSet(
        PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1)
    ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1)
        ))
)


process.l1GtPrescaleFactorsTechTrig = cms.ESProducer("L1GtPrescaleFactorsTechTrigTrivialProducer",
    PrescaleFactorsSet = cms.VPSet(cms.PSet(
        PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1)
    ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1)
        ), 
        cms.PSet(
            PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1)
        ))
)


process.l1GtPsbSetup = cms.ESProducer("L1GtPsbSetupTrivialProducer",
    PsbSetup = cms.VPSet(cms.PSet(
        Slot = cms.int32(9),
        Ch1SendLvds = cms.bool(True),
        Ch0SendLvds = cms.bool(True),
        EnableRecSerLink = cms.vuint32(0, 0, 0, 0, 0, 
            0, 0, 0),
        EnableRecLvds = cms.vuint32(1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 
            1)
    ), 
        cms.PSet(
            Slot = cms.int32(13),
            Ch1SendLvds = cms.bool(False),
            Ch0SendLvds = cms.bool(False),
            EnableRecSerLink = cms.vuint32(1, 1, 1, 1, 1, 
                1, 1, 1),
            EnableRecLvds = cms.vuint32(0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 
                0)
        ), 
        cms.PSet(
            Slot = cms.int32(14),
            Ch1SendLvds = cms.bool(False),
            Ch0SendLvds = cms.bool(False),
            EnableRecSerLink = cms.vuint32(1, 1, 1, 1, 1, 
                1, 1, 1),
            EnableRecLvds = cms.vuint32(0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 
                0)
        ), 
        cms.PSet(
            Slot = cms.int32(15),
            Ch1SendLvds = cms.bool(True),
            Ch0SendLvds = cms.bool(True),
            EnableRecSerLink = cms.vuint32(0, 0, 0, 0, 0, 
                0, 0, 0),
            EnableRecLvds = cms.vuint32(1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 
                1)
        ), 
        cms.PSet(
            Slot = cms.int32(19),
            Ch1SendLvds = cms.bool(False),
            Ch0SendLvds = cms.bool(False),
            EnableRecSerLink = cms.vuint32(0, 0, 0, 0, 0, 
                0, 0, 0),
            EnableRecLvds = cms.vuint32(0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 
                0)
        ), 
        cms.PSet(
            Slot = cms.int32(20),
            Ch1SendLvds = cms.bool(False),
            Ch0SendLvds = cms.bool(False),
            EnableRecSerLink = cms.vuint32(0, 0, 0, 0, 0, 
                0, 0, 0),
            EnableRecLvds = cms.vuint32(0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 
                0)
        ), 
        cms.PSet(
            Slot = cms.int32(21),
            Ch1SendLvds = cms.bool(False),
            Ch0SendLvds = cms.bool(False),
            EnableRecSerLink = cms.vuint32(0, 0, 0, 0, 0, 
                0, 0, 0),
            EnableRecLvds = cms.vuint32(0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 
                0)
        ))
)


process.l1GtStableParameters = cms.ESProducer("L1GtStableParametersTrivialProducer",
    NumberL1IsoEG = cms.uint32(4),
    NumberL1JetCounts = cms.uint32(12),
    UnitLength = cms.int32(8),
    NumberL1ForJet = cms.uint32(4),
    IfCaloEtaNumberBits = cms.uint32(4),
    IfMuEtaNumberBits = cms.uint32(6),
    NumberL1TauJet = cms.uint32(4),
    NumberPsbBoards = cms.int32(7),
    NumberConditionChips = cms.uint32(2),
    NumberL1Mu = cms.uint32(4),
    NumberL1CenJet = cms.uint32(4),
    NumberPhysTriggers = cms.uint32(128),
    PinsOnConditionChip = cms.uint32(96),
    NumberTechnicalTriggers = cms.uint32(64),
    OrderConditionChip = cms.vint32(2, 1),
    NumberPhysTriggersExtended = cms.uint32(64),
    WordLength = cms.int32(64),
    NumberL1NoIsoEG = cms.uint32(4)
)


process.l1GtTriggerMaskAlgoTrig = cms.ESProducer("L1GtTriggerMaskAlgoTrigTrivialProducer",
    TriggerMask = cms.vuint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0)
)


process.l1GtTriggerMaskTechTrig = cms.ESProducer("L1GtTriggerMaskTechTrigTrivialProducer",
    TriggerMask = cms.vuint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0)
)


process.l1GtTriggerMaskVetoAlgoTrig = cms.ESProducer("L1GtTriggerMaskVetoAlgoTrigTrivialProducer",
    TriggerMask = cms.vuint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0)
)


process.l1GtTriggerMaskVetoTechTrig = cms.ESProducer("L1GtTriggerMaskVetoTechTrigTrivialProducer",
    TriggerMask = cms.vuint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0)
)


process.l1GtTriggerMenuXml = cms.ESProducer("L1GtTriggerMenuXmlProducer",
    VmeXmlFile = cms.string(''),
    DefXmlFile = cms.string('L1Menu_Commissioning2009_v1_L1T_Scales_20080926_startup_Imp0.xml'),
    TriggerMenuLuminosity = cms.string('startup')
)


process.trackerGeometry = cms.ESProducer("TrackerDigiGeometryESModule",
    appendToDataLabel = cms.string(''),
    fromDDD = cms.bool(True),
    trackerGeometryConstants = cms.PSet(
        ROCS_X = cms.int32(0),
        ROCS_Y = cms.int32(0),
        upgradeGeometry = cms.bool(False),
        BIG_PIX_PER_ROC_Y = cms.int32(2),
        BIG_PIX_PER_ROC_X = cms.int32(1),
        ROWS_PER_ROC = cms.int32(80),
        COLS_PER_ROC = cms.int32(52)
    ),
    applyAlignment = cms.bool(True),
    alignmentsLabel = cms.string('')
)


process.trackerNumberingGeometry = cms.ESProducer("TrackerGeometricDetESModule",
    appendToDataLabel = cms.string(''),
    fromDDD = cms.bool(True),
    layerNumberPXB = cms.uint32(16),
    totalBlade = cms.uint32(24)
)


process.trackerTopologyConstants = cms.ESProducer("TrackerTopologyEP",
    tob_rodStartBit = cms.uint32(5),
    tib_str_int_extStartBit = cms.uint32(10),
    tib_layerMask = cms.uint32(7),
    pxf_bladeMask = cms.uint32(63),
    appendToDataLabel = cms.string(''),
    pxb_ladderStartBit = cms.uint32(8),
    pxb_layerStartBit = cms.uint32(16),
    tec_wheelStartBit = cms.uint32(14),
    tib_str_int_extMask = cms.uint32(3),
    tec_ringStartBit = cms.uint32(5),
    tib_moduleStartBit = cms.uint32(2),
    tib_sterMask = cms.uint32(3),
    tid_sideStartBit = cms.uint32(13),
    tid_module_fw_bwStartBit = cms.uint32(7),
    tid_ringMask = cms.uint32(3),
    tob_sterMask = cms.uint32(3),
    tec_petal_fw_bwStartBit = cms.uint32(12),
    tec_ringMask = cms.uint32(7),
    tib_strMask = cms.uint32(63),
    tec_sterMask = cms.uint32(3),
    tec_wheelMask = cms.uint32(15),
    tec_sideStartBit = cms.uint32(18),
    pxb_moduleMask = cms.uint32(63),
    pxf_panelStartBit = cms.uint32(8),
    tid_sideMask = cms.uint32(3),
    tob_moduleMask = cms.uint32(7),
    tid_ringStartBit = cms.uint32(9),
    pxf_sideMask = cms.uint32(3),
    pxb_moduleStartBit = cms.uint32(2),
    pxf_diskStartBit = cms.uint32(16),
    tib_str_fw_bwMask = cms.uint32(3),
    tec_moduleMask = cms.uint32(7),
    tid_sterMask = cms.uint32(3),
    tob_rod_fw_bwMask = cms.uint32(3),
    tob_layerStartBit = cms.uint32(14),
    tec_petal_fw_bwMask = cms.uint32(3),
    tib_strStartBit = cms.uint32(4),
    tec_sterStartBit = cms.uint32(0),
    tid_moduleMask = cms.uint32(31),
    tib_sterStartBit = cms.uint32(0),
    tid_sterStartBit = cms.uint32(0),
    pxf_moduleStartBit = cms.uint32(2),
    pxf_diskMask = cms.uint32(15),
    tob_moduleStartBit = cms.uint32(2),
    tid_wheelStartBit = cms.uint32(11),
    tob_layerMask = cms.uint32(7),
    tid_module_fw_bwMask = cms.uint32(3),
    tob_rod_fw_bwStartBit = cms.uint32(12),
    tec_petalMask = cms.uint32(15),
    pxb_ladderMask = cms.uint32(255),
    tec_moduleStartBit = cms.uint32(2),
    tob_rodMask = cms.uint32(127),
    tec_sideMask = cms.uint32(3),
    pxf_sideStartBit = cms.uint32(23),
    pxb_layerMask = cms.uint32(15),
    tib_layerStartBit = cms.uint32(14),
    pxf_panelMask = cms.uint32(3),
    tib_moduleMask = cms.uint32(3),
    pxf_bladeStartBit = cms.uint32(10),
    tid_wheelMask = cms.uint32(3),
    tob_sterStartBit = cms.uint32(0),
    tid_moduleStartBit = cms.uint32(2),
    tec_petalStartBit = cms.uint32(8),
    tib_str_fw_bwStartBit = cms.uint32(12),
    pxf_moduleMask = cms.uint32(63)
)


process.HepPDTESSource = cms.ESSource("HepPDTESSource",
    pdtFileName = cms.FileInPath('SimGeneral/HepPDTESSource/data/pythiaparticle.tbl')
)


process.L1GtBoardMapsRcdSource = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('L1GtBoardMapsRcd'),
    firstValid = cms.vuint32(1)
)


process.L1GtParametersRcdSource = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('L1GtParametersRcd'),
    firstValid = cms.vuint32(1)
)


process.L1GtPrescaleFactorsAlgoTrigRcdSource = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('L1GtPrescaleFactorsAlgoTrigRcd'),
    firstValid = cms.vuint32(1)
)


process.L1GtPrescaleFactorsTechTrigRcdSource = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('L1GtPrescaleFactorsTechTrigRcd'),
    firstValid = cms.vuint32(1)
)


process.L1GtPsbSetupRcdSource = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('L1GtPsbSetupRcd'),
    firstValid = cms.vuint32(1)
)


process.L1GtStableParametersRcdSource = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('L1GtStableParametersRcd'),
    firstValid = cms.vuint32(1)
)


process.L1GtTriggerMaskAlgoTrigRcdSource = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('L1GtTriggerMaskAlgoTrigRcd'),
    firstValid = cms.vuint32(1)
)


process.L1GtTriggerMaskTechTrigRcdSource = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('L1GtTriggerMaskTechTrigRcd'),
    firstValid = cms.vuint32(1)
)


process.L1GtTriggerMaskVetoAlgoTrigRcdSource = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('L1GtTriggerMaskVetoAlgoTrigRcd'),
    firstValid = cms.vuint32(1)
)


process.L1GtTriggerMaskVetoTechTrigRcdSource = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('L1GtTriggerMaskVetoTechTrigRcd'),
    firstValid = cms.vuint32(1)
)


process.L1GtTriggerMenuRcdSource = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('L1GtTriggerMenuRcd'),
    firstValid = cms.vuint32(1)
)


process.XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMother.xml', 
        'Geometry/CMSCommonData/data/cmsTracker.xml', 
        'Geometry/CMSCommonData/data/caloBase.xml', 
        'Geometry/CMSCommonData/data/cmsCalo.xml', 
        'Geometry/CMSCommonData/data/muonBase.xml', 
        'Geometry/CMSCommonData/data/cmsMuon.xml', 
        'Geometry/CMSCommonData/data/mgnt.xml', 
        'Geometry/CMSCommonData/data/beampipe.xml', 
        'Geometry/CMSCommonData/data/cmsBeam.xml', 
        'Geometry/CMSCommonData/data/muonMB.xml', 
        'Geometry/CMSCommonData/data/muonMagnet.xml', 
        'Geometry/TrackerCommonData/data/pixfwdMaterials.xml', 
        'Geometry/TrackerCommonData/data/pixfwdCommon.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq1x2.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq1x5.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x3.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x4.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x5.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPanelBase.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPanel.xml', 
        'Geometry/TrackerCommonData/data/pixfwdBlade.xml', 
        'Geometry/TrackerCommonData/data/pixfwdNipple.xml', 
        'Geometry/TrackerCommonData/data/pixfwdDisk.xml', 
        'Geometry/TrackerCommonData/data/pixfwdCylinder.xml', 
        'Geometry/TrackerCommonData/data/pixfwd.xml', 
        'Geometry/TrackerCommonData/data/pixbarmaterial.xml', 
        'Geometry/TrackerCommonData/data/pixbarladder.xml', 
        'Geometry/TrackerCommonData/data/pixbarladderfull.xml', 
        'Geometry/TrackerCommonData/data/pixbarladderhalf.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer0.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer1.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer2.xml', 
        'Geometry/TrackerCommonData/data/pixbar.xml', 
        'Geometry/TrackerCommonData/data/tibtidcommonmaterial.xml', 
        'Geometry/TrackerCommonData/data/tibmaterial.xml', 
        'Geometry/TrackerCommonData/data/tibmodpar.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0a.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0b.xml', 
        'Geometry/TrackerCommonData/data/tibmodule2.xml', 
        'Geometry/TrackerCommonData/data/tibstringpar.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring0lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring0.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring1lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring1.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring2lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring2.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring3lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring3.xml', 
        'Geometry/TrackerCommonData/data/tiblayerpar.xml', 
        'Geometry/TrackerCommonData/data/tiblayer0.xml', 
        'Geometry/TrackerCommonData/data/tiblayer1.xml', 
        'Geometry/TrackerCommonData/data/tiblayer2.xml', 
        'Geometry/TrackerCommonData/data/tiblayer3.xml', 
        'Geometry/TrackerCommonData/data/tib.xml', 
        'Geometry/TrackerCommonData/data/tidmaterial.xml', 
        'Geometry/TrackerCommonData/data/tidmodpar.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0r.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0l.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1r.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1l.xml', 
        'Geometry/TrackerCommonData/data/tidmodule2.xml', 
        'Geometry/TrackerCommonData/data/tidringpar.xml', 
        'Geometry/TrackerCommonData/data/tidring0.xml', 
        'Geometry/TrackerCommonData/data/tidring0f.xml', 
        'Geometry/TrackerCommonData/data/tidring0b.xml', 
        'Geometry/TrackerCommonData/data/tidring1.xml', 
        'Geometry/TrackerCommonData/data/tidring1f.xml', 
        'Geometry/TrackerCommonData/data/tidring1b.xml', 
        'Geometry/TrackerCommonData/data/tidring2.xml', 
        'Geometry/TrackerCommonData/data/tid.xml', 
        'Geometry/TrackerCommonData/data/tidf.xml', 
        'Geometry/TrackerCommonData/data/tidb.xml', 
        'Geometry/TrackerCommonData/data/tibtidservices.xml', 
        'Geometry/TrackerCommonData/data/tibtidservicesf.xml', 
        'Geometry/TrackerCommonData/data/tibtidservicesb.xml', 
        'Geometry/TrackerCommonData/data/tobmaterial.xml', 
        'Geometry/TrackerCommonData/data/tobmodpar.xml', 
        'Geometry/TrackerCommonData/data/tobmodule0.xml', 
        'Geometry/TrackerCommonData/data/tobmodule2.xml', 
        'Geometry/TrackerCommonData/data/tobmodule4.xml', 
        'Geometry/TrackerCommonData/data/tobrodpar.xml', 
        'Geometry/TrackerCommonData/data/tobrod0c.xml', 
        'Geometry/TrackerCommonData/data/tobrod0l.xml', 
        'Geometry/TrackerCommonData/data/tobrod0h.xml', 
        'Geometry/TrackerCommonData/data/tobrod0.xml', 
        'Geometry/TrackerCommonData/data/tobrod1l.xml', 
        'Geometry/TrackerCommonData/data/tobrod1h.xml', 
        'Geometry/TrackerCommonData/data/tobrod1.xml', 
        'Geometry/TrackerCommonData/data/tobrod2c.xml', 
        'Geometry/TrackerCommonData/data/tobrod2l.xml', 
        'Geometry/TrackerCommonData/data/tobrod2h.xml', 
        'Geometry/TrackerCommonData/data/tobrod2.xml', 
        'Geometry/TrackerCommonData/data/tobrod3l.xml', 
        'Geometry/TrackerCommonData/data/tobrod3h.xml', 
        'Geometry/TrackerCommonData/data/tobrod3.xml', 
        'Geometry/TrackerCommonData/data/tobrod4c.xml', 
        'Geometry/TrackerCommonData/data/tobrod4l.xml', 
        'Geometry/TrackerCommonData/data/tobrod4h.xml', 
        'Geometry/TrackerCommonData/data/tobrod4.xml', 
        'Geometry/TrackerCommonData/data/tobrod5l.xml', 
        'Geometry/TrackerCommonData/data/tobrod5h.xml', 
        'Geometry/TrackerCommonData/data/tobrod5.xml', 
        'Geometry/TrackerCommonData/data/tob.xml', 
        'Geometry/TrackerCommonData/data/tecmaterial.xml', 
        'Geometry/TrackerCommonData/data/tecmodpar.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule2.xml', 
        'Geometry/TrackerCommonData/data/tecmodule3.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule5.xml', 
        'Geometry/TrackerCommonData/data/tecmodule6.xml', 
        'Geometry/TrackerCommonData/data/tecpetpar.xml', 
        'Geometry/TrackerCommonData/data/tecring0.xml', 
        'Geometry/TrackerCommonData/data/tecring1.xml', 
        'Geometry/TrackerCommonData/data/tecring2.xml', 
        'Geometry/TrackerCommonData/data/tecring3.xml', 
        'Geometry/TrackerCommonData/data/tecring4.xml', 
        'Geometry/TrackerCommonData/data/tecring5.xml', 
        'Geometry/TrackerCommonData/data/tecring6.xml', 
        'Geometry/TrackerCommonData/data/tecring0f.xml', 
        'Geometry/TrackerCommonData/data/tecring1f.xml', 
        'Geometry/TrackerCommonData/data/tecring2f.xml', 
        'Geometry/TrackerCommonData/data/tecring3f.xml', 
        'Geometry/TrackerCommonData/data/tecring4f.xml', 
        'Geometry/TrackerCommonData/data/tecring5f.xml', 
        'Geometry/TrackerCommonData/data/tecring6f.xml', 
        'Geometry/TrackerCommonData/data/tecring0b.xml', 
        'Geometry/TrackerCommonData/data/tecring1b.xml', 
        'Geometry/TrackerCommonData/data/tecring2b.xml', 
        'Geometry/TrackerCommonData/data/tecring3b.xml', 
        'Geometry/TrackerCommonData/data/tecring4b.xml', 
        'Geometry/TrackerCommonData/data/tecring5b.xml', 
        'Geometry/TrackerCommonData/data/tecring6b.xml', 
        'Geometry/TrackerCommonData/data/tecpetalf.xml', 
        'Geometry/TrackerCommonData/data/tecpetalb.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal6f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal6b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal8f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal8b.xml', 
        'Geometry/TrackerCommonData/data/tecwheel.xml', 
        'Geometry/TrackerCommonData/data/tecwheela.xml', 
        'Geometry/TrackerCommonData/data/tecwheelb.xml', 
        'Geometry/TrackerCommonData/data/tecwheelc.xml', 
        'Geometry/TrackerCommonData/data/tecwheeld.xml', 
        'Geometry/TrackerCommonData/data/tecwheel6.xml', 
        'Geometry/TrackerCommonData/data/tecservices.xml', 
        'Geometry/TrackerCommonData/data/tecbackplate.xml', 
        'Geometry/TrackerCommonData/data/tec.xml', 
        'Geometry/TrackerCommonData/data/trackermaterial.xml', 
        'Geometry/TrackerCommonData/data/tracker.xml', 
        'Geometry/TrackerCommonData/data/trackerpixbar.xml', 
        'Geometry/TrackerCommonData/data/trackerpixfwd.xml', 
        'Geometry/TrackerCommonData/data/trackertibtidservices.xml', 
        'Geometry/TrackerCommonData/data/trackertib.xml', 
        'Geometry/TrackerCommonData/data/trackertid.xml', 
        'Geometry/TrackerCommonData/data/trackertob.xml', 
        'Geometry/TrackerCommonData/data/trackertec.xml', 
        'Geometry/TrackerCommonData/data/trackerbulkhead.xml', 
        'Geometry/TrackerCommonData/data/trackerother.xml', 
        'Geometry/EcalCommonData/data/eregalgo.xml', 
        'Geometry/EcalCommonData/data/ebalgo.xml', 
        'Geometry/EcalCommonData/data/ebcon.xml', 
        'Geometry/EcalCommonData/data/ebrot.xml', 
        'Geometry/EcalCommonData/data/eecon.xml', 
        'Geometry/EcalCommonData/data/eefixed.xml', 
        'Geometry/EcalCommonData/data/eehier.xml', 
        'Geometry/EcalCommonData/data/eealgo.xml', 
        'Geometry/EcalCommonData/data/escon.xml', 
        'Geometry/EcalCommonData/data/esalgo.xml', 
        'Geometry/EcalCommonData/data/eeF.xml', 
        'Geometry/EcalCommonData/data/eeB.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'Geometry/HcalCommonData/data/hcalalgo.xml', 
        'Geometry/HcalCommonData/data/hcalbarrelalgo.xml', 
        'Geometry/HcalCommonData/data/hcalendcapalgo.xml', 
        'Geometry/HcalCommonData/data/hcalouteralgo.xml', 
        'Geometry/HcalCommonData/data/hcalforwardalgo.xml', 
        'Geometry/HcalCommonData/data/average/hcalforwardmaterial.xml', 
        'Geometry/MuonCommonData/data/mbCommon.xml', 
        'Geometry/MuonCommonData/data/mb1.xml', 
        'Geometry/MuonCommonData/data/mb2.xml', 
        'Geometry/MuonCommonData/data/mb3.xml', 
        'Geometry/MuonCommonData/data/mb4.xml', 
        'Geometry/MuonCommonData/data/muonYoke.xml', 
        'Geometry/MuonCommonData/data/mf.xml', 
        'Geometry/ForwardCommonData/data/forward.xml', 
        'Geometry/ForwardCommonData/data/bundle/forwardshield.xml', 
        'Geometry/ForwardCommonData/data/brmrotations.xml', 
        'Geometry/ForwardCommonData/data/brm.xml', 
        'Geometry/ForwardCommonData/data/totemMaterials.xml', 
        'Geometry/ForwardCommonData/data/totemRotations.xml', 
        'Geometry/ForwardCommonData/data/totemt1.xml', 
        'Geometry/ForwardCommonData/data/totemt2.xml', 
        'Geometry/ForwardCommonData/data/ionpump.xml', 
        'Geometry/MuonCommonData/data/muonNumbering.xml', 
        'Geometry/TrackerCommonData/data/trackerStructureTopology.xml', 
        'Geometry/TrackerSimData/data/trackersens.xml', 
        'Geometry/TrackerRecoData/data/trackerRecoMaterial.xml', 
        'Geometry/EcalSimData/data/ecalsens.xml', 
        'Geometry/HcalCommonData/data/hcalsenspmf.xml', 
        'Geometry/HcalSimData/data/hf.xml', 
        'Geometry/HcalSimData/data/hfpmt.xml', 
        'Geometry/HcalSimData/data/hffibrebundle.xml', 
        'Geometry/HcalSimData/data/CaloUtil.xml', 
        'Geometry/MuonSimData/data/muonSens.xml', 
        'Geometry/DTGeometryBuilder/data/dtSpecsFilter.xml', 
        'Geometry/CSCGeometryBuilder/data/cscSpecsFilter.xml', 
        'Geometry/CSCGeometryBuilder/data/cscSpecs.xml', 
        'Geometry/RPCGeometryBuilder/data/RPCSpecs.xml', 
        'Geometry/ForwardCommonData/data/brmsens.xml', 
        'Geometry/HcalSimData/data/HcalProdCuts.xml', 
        'Geometry/EcalSimData/data/EcalProdCuts.xml', 
        'Geometry/EcalSimData/data/ESProdCuts.xml', 
        'Geometry/TrackerSimData/data/trackerProdCuts.xml', 
        'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml', 
        'Geometry/MuonSimData/data/muonProdCuts.xml', 
        'Geometry/ForwardSimData/data/ForwardShieldProdCuts.xml', 
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


process.eegeom = cms.ESSource("EmptyESSource",
    iovIsRunNotTime = cms.bool(True),
    recordName = cms.string('EcalMappingRcd'),
    firstValid = cms.vuint32(1)
)


process.HcalReLabel = cms.PSet(
    RelabelRules = cms.untracked.PSet(
        Eta16 = cms.untracked.vint32(1, 1, 2, 2, 2, 
            2, 2, 2, 2, 3, 
            3, 3, 3, 3, 3, 
            3, 3, 3, 3),
        Eta17 = cms.untracked.vint32(1, 1, 2, 2, 3, 
            3, 3, 4, 4, 4, 
            4, 4, 5, 5, 5, 
            5, 5, 5, 5),
        Eta1 = cms.untracked.vint32(1, 2, 2, 2, 3, 
            3, 3, 3, 3, 3, 
            3, 3, 3, 3, 3, 
            3, 3, 3, 3),
        CorrectPhi = cms.untracked.bool(False)
    ),
    RelabelHits = cms.untracked.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    SkipEvent = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)


