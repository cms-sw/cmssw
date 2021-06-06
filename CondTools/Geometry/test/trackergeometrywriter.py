import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerGeometryWriter")
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.Geometry.GeometryExtended_cff')

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.TrackerGeometricDetESModule = cms.ESProducer( "TrackerGeometricDetESModule",
                                                      fromDDD = cms.bool( True )
                                                     )

process.TrackerGeometryWriter = cms.EDAnalyzer("PGeometricDetBuilder", fromDD4hep=cms.bool(False))
process.TrackerParametersWriter = cms.EDAnalyzer("PTrackerParametersDBBuilder", fromDD4hep=cms.bool(False))

process.CondDB.timetype = cms.untracked.string('runnumber')
process.CondDB.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('IdealGeometryRecord'),tag = cms.string('TKRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PTrackerParametersRcd'),tag = cms.string('TKParameters_Geometry_Test01'))
                                                            )
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.TrackerGeometryWriter*process.TrackerParametersWriter)
