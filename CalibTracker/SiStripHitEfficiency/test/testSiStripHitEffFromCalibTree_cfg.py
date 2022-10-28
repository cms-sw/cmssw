import FWCore.ParameterSet.Config as cms

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing()
options.register("inputFiles", "", VarParsing.multiplicity.list, VarParsing.varType.string, "Files to process")
options.register("runNumber", 0, VarParsing.multiplicity.singleton, VarParsing.varType.int, "Run number")
options.parseArguments()

process = cms.Process("HitEff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')  

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(options.runNumber),
    lastValue = cms.uint64(options.runNumber),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

process.SiStripHitEff = cms.EDAnalyzer("SiStripHitEffFromCalibTree",
    CalibTreeFilenames = cms.untracked.vstring(options.inputFiles),
    Threshold         = cms.double(0.1),
    nModsMin          = cms.int32(5),
    doSummary         = cms.int32(0),
    #ResXSig           = cms.untracked.double(5),
    SinceAppendMode   = cms.bool(True),
    IOVMode           = cms.string('Run'),
    Record            = cms.string('SiStripBadStrip'),
    doStoreOnDB       = cms.bool(True),
    BadModulesFile    = cms.untracked.string("BadModules_input.txt"),   # default "" no input
    AutoIneffModTagging = cms.untracked.bool(True),   # default true, automatic limit for each layer to identify inefficient modules
    ClusterMatchingMethod  = cms.untracked.int32(4),     # default 0  case0,1,2,3,4
    ClusterTrajDist   = cms.untracked.double(15),   # default 64
    StripsApvEdge     = cms.untracked.double(10),   # default 10  
    UseOnlyHighPurityTracks = cms.untracked.bool(True), # default True
    SpaceBetweenTrains = cms.untracked.int32(25),   # default 25
    ShowEndcapSides   = cms.untracked.bool(True),  # default True
    ShowRings         = cms.untracked.bool(False),  # default False
    showTOB6TEC9      = cms.untracked.bool(False),  # default False
    TkMapMin          = cms.untracked.double(0.90), # default 0.90
    EffPlotMin        = cms.untracked.double(0.90), # default 0.90
    Title             = cms.string(' Hit Efficiency - run {0:d}'.format(options.runNumber))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStrip'),
        tag = cms.string('SiStripHitEffBadModules')
    ))
)

process.TFileService = cms.Service("TFileService",
        fileName = cms.string('SiStripHitEffHistos_run{0:d}.root'.format(options.runNumber))
)

process.allPath = cms.Path(process.SiStripHitEff)
