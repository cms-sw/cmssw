import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/giordano/TIF_reconstruction_RealData_TIBTOB_full.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStrip'),
        tag = cms.string('SiStripBadChannel_v1')
    ))
)

process.prod = cms.EDFilter("SiStripQualityHotStripIdentifier",
    Track_src = cms.untracked.InputTag("ctfWithMaterialTracksTIFTIBTOB"),
    AlgoParameters = cms.PSet(
        AlgoName = cms.string('SiStripHotStripAlgorithmFromClusterOccupancy'),
        MinNumEntries = cms.untracked.uint32(100),
        ProbabilityThreshold = cms.untracked.double(1e-07),
        MinNumEntriesPerStrip = cms.untracked.uint32(10)
    ),
    SinceAppendMode = cms.bool(True),
    IOVMode = cms.string('Run'),
    Record = cms.string('SiStripBadStrip'),
    Cluster_src = cms.InputTag("siStripClusters"),
    doStoreOnDB = cms.bool(True),
    ClusterSelection = cms.untracked.PSet(
        minWidth = cms.untracked.uint32(1),
        maxWidth = cms.untracked.uint32(10000)
    ),
    RemoveTrackClusters = cms.untracked.bool(True)
)

process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.print)


