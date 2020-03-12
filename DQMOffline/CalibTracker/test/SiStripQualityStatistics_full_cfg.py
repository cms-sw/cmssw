import FWCore.ParameterSet.Config as cms

process = cms.Process("CALIB")
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
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
        tag = cms.string('SiStripBadChannel_v1')
    )),
    connect = cms.string('sqlite_file:dbfile.db')
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
    ), 
        cms.PSet(
            record = cms.string('SiStripBadChannelRcd'),
            tag = cms.string('SiStripBadChannel_GR_21X_v2_hlt')
        )),
    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_21X_STRIP')
)

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.SiStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
    PrintDebug = cms.untracked.bool(True),
    appendToDataLabel = cms.string('test'),
    ReduceGranularity = cms.bool(True),
    ThresholdForReducedGranularity = cms.double(0.3),
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadModuleRcd'),
        tag = cms.string('')
    ), 
        cms.PSet(
            record = cms.string('SiStripBadChannelRcd'),
            tag = cms.string('')
        ), 
        cms.PSet(
            record = cms.string('SiStripDetCablingRcd'),
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
                             #TkMapFileName = cms.untracked.string('TkMaps/TkMapBadComponents_full.png'),
                             TkMapFileName = cms.untracked.string(''),
                             dataLabel = cms.untracked.string('test')
                             )

process.out = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.stat)
process.ep = cms.EndPath(process.out)

