import FWCore.ParameterSet.Config as cms

process = cms.Process("BuildPPSGeometry")

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# output database (in this case local sqlite file)
process.CondDB.connect = 'sqlite_file:PPSGeometry_2017.db'

# We define the output service.
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('VeryForwardIdealGeometryRecord'),
        tag = cms.string('PPSGeometry_test')
    ))
)

# minimum of logs
#process.MessageLogger = cms.Service("MessageLogger",
#    statistics = cms.untracked.vstring(),
#    destinations = cms.untracked.vstring('cout'),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('INFO')
#    )
#)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('ppsGeometryBuilder')
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('DEBUG'))


# geometry
#from Geometry.VeryForwardGeometry.geometryIdealPPSFromDD_2017_cfi import allFiles

#process.XMLIdealGeometryESSource_CTPPS = cms.ESSource("XMLIdealGeometryESSource",
#    geomXMLFiles = allFiles,
#    rootNodeName = cms.string('cms:CMSE')
#)

# dd4hep-based geometry
process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/VeryForwardGeometry/test/geometryIdealPPSFromDD_2018_ddhep.xml'),
                                            #confGeomXMLFiles = cms.FileInPath('Geometry/VeryForwardGeometry/data/dd4hep/cms-pps-reco-geometry-2018.xml'),
                                            appendToDataLabel = cms.string('XMLIdealGeometryESSource_CTPPS')
)

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                 appendToDataLabel = cms.string('XMLIdealGeometryESSource_CTPPS')
)


# no events to process
process.source = cms.Source("EmptySource",
  firstRun = cms.untracked.uint32(123),
  firstLuminosityBlock = cms.untracked.uint32(1),
  firstEvent = cms.untracked.uint32(1),
  numberEventsInLuminosityBlock = cms.untracked.uint32(3),
  numberEventsInRun = cms.untracked.uint32(30)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.ppsGeometryBuilder = cms.EDAnalyzer("PPSGeometryBuilder",
    compactViewTag = cms.untracked.string('XMLIdealGeometryESSource_CTPPS')
)

process.p = cms.Path(
    process.ppsGeometryBuilder
)
