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
process.b = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripFedCablingRcd'),
        tag = cms.string('SiStripFedCabling_GR_21X_v2_hlt')
    )),
    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_21X_STRIP')
)

process.TkDetMap = cms.Service("TkDetMap")

process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.prod = cms.EDAnalyzer("SiStripDQMProfileToTkMapConverter",
    TkMapFileName = cms.untracked.string('CabTkMaptest_insertRun.png'),
    verbosity = cms.untracked.uint32(0),
    rootFilename = cms.untracked.string('insertFile'),
    rootDirPath = cms.untracked.string('Run insertRun/SiStrip'),
    TkMapDrawOption = cms.untracked.string('Zcol')
)

process.p = cms.Path(process.prod)


