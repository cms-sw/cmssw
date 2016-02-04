import FWCore.ParameterSet.Config as cms

process = cms.Process("DetVOffReader")

process.MessageLogger = cms.Service(
    "MessageLogger",
    debugModules = cms.untracked.vstring(''),
    threshold = cms.untracked.string('INFO'),
    destinations = cms.untracked.vstring('SiStripDetVOffReader.log')
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripDetVOffRcd'),
        tag = cms.string('SiStripDetVOff_v1')
    ))
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.TrackerDigiGeometryESModule.applyAlignment = False

process.fedcablingreader = cms.EDAnalyzer("SiStripDetVOffReader")

process.p1 = cms.Path(process.fedcablingreader)


