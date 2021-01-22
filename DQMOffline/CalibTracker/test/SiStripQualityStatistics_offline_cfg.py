import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(100000),
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
        tag = cms.string('inputTag')
    )),
    connect = cms.string('sqlite_file:dbfile.db')
)

process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
    PrintDebug = cms.untracked.bool(True),
    appendToDataLabel = cms.string('test'),
    ReduceGranularity = cms.bool(True),
    ThresholdForReducedGranularity = cms.double(0.3),
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadModuleRcd'),
        tag = cms.string('')
    ))
)

#### Add these lines to produce a tracker map
#process.load("DQM.SiStripCommon.TkHistoMap_cff")
### load TrackerTopology (needed for TkDetMap and TkHistoMap)
#process.load("Configuration.Geometry.GeometryExtended2017_cff")
#process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")
#process.trackerTopology = cms.ESProducer("TrackerTopologyEP")
####

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.stat = DQMEDAnalyzer("SiStripQualityStatistics",
                             #TkMapFileName = cms.untracked.string('TkMaps/TkMapBadComponents_offline.png'),
                             TkMapFileName = cms.untracked.string(''),
                             dataLabel = cms.untracked.string('test')
                             )

process.out = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.stat)
process.ep = cms.EndPath(process.out)

