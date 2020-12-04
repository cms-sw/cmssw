import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(110213),
    lastRun = cms.untracked.uint32(110213),
    interval = cms.uint32(1)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        SiStripFEDErrorsDQMReader = cms.untracked.PSet(

        )
    ),
    threshold = cms.untracked.string('INFO')
)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string("SiStripBadStripRcd"),
        tag = cms.string("SiStripBadStrip_test1")
    )),
    # connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_STRIP')
    connect = cms.string('sqlite_file:dbfile.db')
)

process.prod = cms.EDAnalyzer("SiStripBadComponentsDQMServiceReader",
                            printDebug = cms.untracked.bool(True)
)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
#process.ep = cms.EndPath(process.print)


