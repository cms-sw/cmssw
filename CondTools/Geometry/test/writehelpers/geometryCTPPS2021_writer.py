import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryWriter")

process.load('CondCore.CondDB.CondDB_cfi')

# geometry
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2021_cfi")

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

# This reads the big XML file and the only way to fill the
# nonreco part of the database is to read this file.  It
# somewhat duplicates the information read from the little
# XML files, but there is no way to directly build the
# DDCompactView from this.
process.XMLGeometryWriter = cms.EDAnalyzer("XMLGeometryBuilder",
                                           XMLFileName = cms.untracked.string("./ge2021SingleBigFile.xml"),
                                           ZIP = cms.untracked.bool(True)
                                           )

# DB writer
process.ppsGeometryBuilder = cms.EDAnalyzer("PPSGeometryBuilder",
                                            fromDD4hep = cms.untracked.bool(False),
                                            isRun2 = cms.untracked.bool(False),
                                            compactViewTag = cms.untracked.string('XMLIdealGeometryESSource_CTPPS')
)

process.CondDB.timetype = cms.untracked.string('runnumber')
process.CondDB.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'),
                                                                     tag = cms.string('XMLFILE_CTPPS_Geometry_2021_TagXX')),
                                                            cms.PSet(record = cms.string('VeryForwardIdealGeometryRecord'),
                                                                     tag = cms.string('PPSRECO_Geometry_2021_TagXX'))
                                                            )
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.XMLGeometryWriter+process.ppsGeometryBuilder)

