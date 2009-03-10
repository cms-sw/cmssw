import FWCore.ParameterSet.Config as cms

process = cms.Process("XMLGeometryWriter")
# empty input service, fire 10 events
#    include "FWCore/MessageLogger/data/MessageLogger.cfi"
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.XMLGeometryWriter = cms.EDAnalyzer("XMLGeometryBuilder",
                                           XMLFileName = cms.untracked.string("./fred.xml"),
                                           ZIP = cms.untracked.bool(True)
                                           )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                          timetype = cms.untracked.string('runnumber'),
                                          connect = cms.string('sqlite_file:myfile.db'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'),tag = cms.string('XMLFILE_Geometry_Test03')))
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.XMLGeometryWriter)

