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

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V3P::All"

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
    OccupancyRootFile = cms.untracked.string('BadAPVOccupancy_insertRun.root'),
    WriteOccupancyRootFile = cms.untracked.bool(True),
    AlgoParameters = cms.PSet(
        AlgoName = cms.string('SiStripBadAPVAlgorithmFromClusterOccupancy'),
        OccupancyHisto = cms.untracked.string('ClusterPosition__det__'),
        LowOccupancyThreshold  = cms.untracked.double(3),
        HighOccupancyThreshold = cms.untracked.double(5),
        AbsoluteLowThreshold   = cms.untracked.double(10),
        NumberIterations = cms.untracked.uint32(3)
    ),
    SinceAppendMode = cms.bool(True),
    verbosity = cms.untracked.uint32(0),
    IOVMode = cms.string('Run'),
    Record = cms.string('SiStripBadStrip'),
    rootDirPath = cms.untracked.string(''),
    rootFilename = cms.untracked.string('insertInputDQMfile'),
    doStoreOnDB = cms.bool(True)
)

process.out = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.out)

