import FWCore.ParameterSet.Config as cms

a = cms.ESSource("PoolDBESSource",
    appendToDataLabel = cms.string('test'),
    toGet = cms.VPSet(
    cms.PSet(
        record = cms.string('SiStripBadFiberRcd'),
        tag = cms.string('insert_DB_Tag')
    )
    ),
                DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)

siStripQualityESProducer = cms.ESProducer("SiStripQualityESProducer",
    ThresholdForReducedGranularity = cms.double(0.3),
    appendToDataLabel = cms.string(''),
    ReduceGranularity = cms.bool(True),
    ListOfRecordToMerge = cms.VPSet(
    cms.PSet(
        record = cms.string('SiStripBadFiberRcd'),
        tag = cms.string('test')
    )
    )
)

es_prefer_siStripQualityESProducer = cms.ESPrefer("SiStripQualityESProducer","siStripQualityESProducer")


