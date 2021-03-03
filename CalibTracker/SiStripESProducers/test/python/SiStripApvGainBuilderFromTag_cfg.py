import FWCore.ParameterSet.Config as cms

process = cms.Process("APVGAINBUILDER")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    ),
    threshold = cms.untracked.string('INFO')
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
    connect = cms.string('sqlite_file:dbfile_gainFromDataCorrected070.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd2'),
        tag = cms.string('SiStripApvGain_gaussian')
    ))
)

process.load('Configuration.Geometry.GeometryExtended_cff')
process.TrackerTopologyEP = cms.ESProducer("TrackerTopologyEP")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.trackerGeometry.applyAlignment = False

process.prod = cms.EDAnalyzer("SiStripApvGainBuilderFromTag",
                            genMode = cms.string("gaussian"),
                            applyTuning = cms.bool(True),
### used if tuning is requested:
                            # TIB
                            correctTIB = cms.vdouble( 0.985,  1.,  1.,  1.),
                            # TID                         
                            correctTID = cms.vdouble( 0.957,  0.931,  0.971),
                            # TOB                         
                            correctTOB = cms.vdouble( 1.07,  1.08,  1.1,  1.07,  1.135,  1.135),
                            # TEC
                            correctTEC = cms.vdouble( 1.09,  1.075,  1.09,  1.06,  1.095,  1.06,  1.08),
### used if gaussian genMode is requested:                           
                            MeanGain = cms.double(1.),
                            SigmaGain = cms.double(0.07),
                            MinPositiveGain = cms.double(0.)
)

# process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
# process.ep = cms.EndPath(process.print)


