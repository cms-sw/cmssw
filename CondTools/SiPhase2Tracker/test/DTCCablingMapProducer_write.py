import FWCore.ParameterSet.Config as cms

process = cms.Process("DTCCablingMapProducer")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# output database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:OuterTrackerDTCCablingMap.db'

# A data source must always be defined. We don't need it, so here's a dummy one.
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

# We define the output service.
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TrackerDetToDTCELinkCablingMapRcd'),
        tag = cms.string('DTCCablingMapProducerUserRun')
    ))
)

process.otdtccablingmap_producer = cms.EDAnalyzer("DTCCablingMapProducer",
    record = cms.string('TrackerDetToDTCELinkCablingMapRcd'),
    generate_fake_valid_gbtlink_and_elinkid = cms.bool(True),
    modulesToDTCCablingCSVFileNames = cms.vstring(
      "CondTools/SiPhase2Tracker/data/TrackerModuleToDTCCablingMap__OT616_200_IT613__T14__OTOnly.csv"
    ),
    csvFormat_ncolumns = cms.uint32( 2),
    csvFormat_idetid   = cms.uint32( 0),
    csvFormat_idtcid   = cms.uint32( 1),
    verbosity = cms.int32(0),
    #loggingOn= cms.untracked.bool(True),
    #SinceAppendMode=cms.bool(True),
    #Source=cms.PSet(
        #IOVRun=cms.untracked.uint32(1)
    #)
)

process.path = cms.Path(process.otdtccablingmap_producer)
