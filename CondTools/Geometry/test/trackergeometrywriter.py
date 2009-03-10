import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerGeometryWriter")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load('Configuration/StandardSequences/GeometryIdeal_cff')


process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )


process.TrackerGeometryWriter = cms.EDAnalyzer("PGeometricDetBuilder")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),

                                          timetype = cms.untracked.string('runnumber'),
                                          connect = cms.string('sqlite_file:myfile.db'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('IdealGeometryRecord'),tag = cms.string('TKRECO_Geometry_Test02'))
                                                            )
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.TrackerGeometryWriter)

