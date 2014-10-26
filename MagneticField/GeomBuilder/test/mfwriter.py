import FWCore.ParameterSet.Config as cms

process = cms.Process("MagneticFieldWriter")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

# This will read all the little XML files and from
# that fill the DDCompactView. The modules that fill
# the reco part of the database need the DDCompactView.
#process.load('Configuration.Geometry.MagneticFieldGeometry_cff')

#GEOMETRY_VERSION = 90322
#GEOMETRY_VERSION = 120812
GEOMETRY_VERSION = 130503

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
                                           XMLFileName = cms.untracked.string("./mfGeometry_"+str(GEOMETRY_VERSION)+".xml"),
                                           ZIP = cms.untracked.bool(True),
                                           record = cms.untracked.string('MFGeometryFileRcd')
                                           )

process.CondDBCommon.BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
process.CondDBCommon.timetype = cms.untracked.string('runnumber')
process.CondDBCommon.connect = cms.string('sqlite_file:mfGeometry_'+str(GEOMETRY_VERSION)+'.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('MFGeometryFileRcd'),tag = cms.string('MagneticFieldGeometry_'+str(GEOMETRY_VERSION))))
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.XMLGeometryWriter)
