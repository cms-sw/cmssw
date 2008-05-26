import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        BlobStreamerName = cms.untracked.string('DefaultBlobStreamingService')
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripPedestalsRcd'),
        tag = cms.string('SiStripPedNoise_TOB_v1_p')
    )),
    connect = cms.string('oracle://cms_orcoff_int2r/CMS_COND_STRIP')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(10),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripPedestalsRcd'),
        data = cms.vstring('SiStripPedestals')
    )),
    verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.get)


