import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
process.load("DQMServices.Core.DQM_cfg")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(insertRun),
    lastValue = cms.uint64(insertRun),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V3P::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStrip'),
        tag = cms.string('SiStripBadChannel_v1')
    ))
)

process.prod = cms.EDFilter("SiStripQualityHotStripIdentifierRoot",
    OccupancyRootFile = cms.untracked.string('HotStripsOccupancy_insertRun.root'),
    WriteOccupancyRootFile = cms.untracked.bool(True), # Ouput File has a size of ~100MB. To suppress writing set parameter to 'False'
    OccupancyH_Xmax = cms.untracked.double(1.0),
    AlgoParameters = cms.PSet(
        AlgoName = cms.string('SiStripHotStripAlgorithmFromClusterOccupancy'),
        NumberOfEvents = cms.untracked.uint32(0),
        ProbabilityThreshold = cms.untracked.double(1e-07),
        MinNumEntriesPerStrip = cms.untracked.uint32(20),
        MinNumEntries = cms.untracked.uint32(0),
        OccupancyThreshold = cms.untracked.double(0.0001)
    ),
    SinceAppendMode = cms.bool(True),
    verbosity = cms.untracked.uint32(0),
    OccupancyH_Xmin = cms.untracked.double(-0.0005),
    IOVMode = cms.string('Run'),
    Record = cms.string('SiStripBadStrip'),
    rootDirPath = cms.untracked.string(''),
    rootFilename = cms.untracked.string('insertInputDQMfile'),
    doStoreOnDB = cms.bool(True),
    OccupancyH_Nbin = cms.untracked.uint32(1001)
)

process.out = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.out)

