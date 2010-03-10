import FWCore.ParameterSet.Config as cms

process = cms.Process("APVGAINBUILDER")
process.MessageLogger = cms.Service("MessageLogger",
    threshold = cms.untracked.string('INFO'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(128408)
)

# process.source = cms.Source("EmptyIOVSource",
#     firstValue = cms.uint64(128409),
#     lastValue = cms.uint64(128409),
#     timetype = cms.string('runnumber'),
#     interval = cms.uint64(1)
# )

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
    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_STRIP'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SiStripApvGain_GR10_v1_hlt')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd2'),
        tag = cms.string('SiStripApvGain_gaussian')
    ))
)

process.prod = cms.EDFilter("SiStripApvGainBuilderFromTag",
                            genMode = cms.string("gaussian"),
                            MeanGain = cms.double(1.),
                            SigmaGain = cms.double(0.1),
                            MinPositiveGain = cms.double(0.)
)

# process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
# process.ep = cms.EndPath(process.print)


