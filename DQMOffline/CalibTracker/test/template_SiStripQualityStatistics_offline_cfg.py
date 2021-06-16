import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")
process.MessageLogger = cms.Service("MessageLogger",
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
    input = cms.untracked.int32(-1)
)
process.a = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadModuleRcd'),
        tag = cms.string('SiStripHotAPVs')
    ),
                      cms.PSet(
        record = cms.string('SiStripBadFiberRcd'),
        tag = cms.string('SiStripHotStrips')
    )),
    connect = cms.string('sqlite_file:dbfile.db')
)

process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
    PrintDebug = cms.untracked.bool(True),
    PrintDebugOutput = cms.bool(False),
    UseEmptyRunInfo = cms.bool(False),
    appendToDataLabel = cms.string('test'),
    ReduceGranularity = cms.bool(False),
    ThresholdForReducedGranularity = cms.double(0.3),
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadModuleRcd'),
        tag = cms.string('')
    ),
                                    cms.PSet(
        record = cms.string('SiStripBadFiberRcd'),
        tag = cms.string('')
    ))
)

#### Add these lines to produce a tracker map
process.load("DQM.SiStripCommon.TkHistoMap_cfi")
# load TrackerTopology (needed for TkDetMap and TkHistoMap)
process.load("Geometry.CMSCommonData.cmsExtendedGeometry2017XML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
process.trackerTopology = cms.ESProducer("TrackerTopologyEP")
####

from CalibTracker.SiStripQuality.siStripQualityStatistics_cfi import siStripQualityStatistics
process.stat = siStripQualityStatistics.clone(
        TkMapFileName = cms.untracked.string('TkMapBadComponents_offline.png'),
        StripQualityLabel = cms.string("test")
        )

process.out = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.stat)
process.ep = cms.EndPath(process.out)

