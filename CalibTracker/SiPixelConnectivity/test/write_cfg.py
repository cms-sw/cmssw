import FWCore.ParameterSet.Config as cms

process = cms.Process("MapWriter")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string("sqlite_file:cabling.db")

#process.CondDBCommon.connect = cms.InputTag("sqlite_file","cabling.db")
#process.CondDBCommon.catalog = cms.InputTag("file","testcatalog.xml")

#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
#process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FakeConditions_cff")


#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('IDEAL_V9::All')
#process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')


#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    toPut = cms.VPSet(cms.PSet(
        record =  cms.string('SiPixelFedCablingMapRcd'),
        tag = cms.string('SiPixelFedCablingMap_v14')
    )),
    loadBlobStreamer = cms.untracked.bool(False)
)

#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('*'),
#    destinations = cms.untracked.vstring('out'),
#    out = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG'))
#)

process.load("CalibTracker.SiPixelConnectivity.PixelToLNKAssociateFromAsciiESProducer_cfi")

process.mapwriter = cms.EDFilter("SiPixelFedCablingMapWriter",
  record = cms.string('SiPixelFedCablingMapRcd'),
  associator = cms.untracked.string('PixelToLNKAssociateFromAscii')
)

process.p1 = cms.Path(process.mapwriter)
