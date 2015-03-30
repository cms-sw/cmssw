import FWCore.ParameterSet.Config as cms


process = cms.Process("ICALIB")
process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(235679),
    lastValue = cms.uint64(235679),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)



process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.destinations = cms.untracked.vstring('cout')
process.MessageLogger.cout = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG'))

#process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.Geometry.GeometryIdeal_cff')

#Setup the SiSTripFedCabling and the SiStripDetCabling
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect='frontier://FrontierProd/CMS_COND_31X_STRIP'

process.poolDBESSource = cms.ESSource( 'PoolDBESSource',
                                       process.CondDBCommon,
                                       BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                       toGet = cms.VPSet( 
                                                          cms.PSet( record = cms.string('SiStripFedCablingRcd'),
                                                                    tag    = cms.string('SiStripFedCabling_GR10_v1_hlt')
                                                                  )
                                                        )
                                     )
                                                                    
process.load("CalibTracker.SiStripESProducers.SiStripConnectivity_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SiStripApvGainRcd_v1')
    ))
)

process.prod = cms.EDAnalyzer("SiStripApvGainFromFileBuilder",
    outputMaps    = cms.untracked.bool(True),
    outputSummary = cms.untracked.bool(True),
)

process.p = cms.Path(process.prod)


