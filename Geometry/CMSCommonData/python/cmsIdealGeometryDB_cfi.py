import FWCore.ParameterSet.Config as cms

PoolDBESSource = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('IdealGeometryRecord'),
        tag = cms.string('IdealGeometry01')
    )),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    catalog = cms.untracked.string('file:PoolFileCatalog.xml'),
    timetype = cms.string('runnumber'),
    connect = cms.string('sqlite_file:testIdeal.db')
)


