import FWCore.ParameterSet.Config as cms

process = cms.Process("NumberingTest")
# empty input service, fire 10 events
#    include "FWCore/MessageLogger/data/MessageLogger.cfi"
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")

# Choose Tracker Geometry
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.source = cms.Source("EmptyIOVSource",
    lastRun = cms.untracked.uint32(1),
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(1),
    interval = cms.uint32(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PGeometricDetRcd'),
        tag = cms.string('PGeometricDet')
    )),
    loadBlobStreamer = cms.untracked.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.prod = cms.EDFilter("GeometricDetLoader")

process.p1 = cms.Path(process.prod)
process.CondDBCommon.connect = cms.InputTag("sqlite_file","test.db")
process.CondDBCommon.catalog = cms.InputTag("file","testcatalog.xml")

