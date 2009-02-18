import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    lastRun = cms.untracked.uint32(1),
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(1),
    interval = cms.uint32(1)
)


process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo.root")
                                   )


process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.Timing = cms.Service("Timing")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.GeometryIdeal_cff")

process.QualityReader = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelQualityRcd'),
        tag = cms.string('SiPixelBadModule_test')
    )),
    connect = cms.string('sqlite_file:test.db')
)

process.es_prefer_QualityReader = cms.ESPrefer("PoolDBESSource","QualityReader")

process.BadModuleReader = cms.EDFilter("SiPixelBadModuleReader",
    printDebug = cms.untracked.uint32(0)
)

process.p = cms.Path(process.BadModuleReader)



