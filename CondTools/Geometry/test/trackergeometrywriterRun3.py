import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerGeometryWriter")
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.Geometry.GeometryExtended2023sim_cff')

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.TrackerGeometricDetExtraESModule = cms.ESProducer( "TrackerGeometricDetExtraESModule",
                                                           fromDDD = cms.bool( True )
                                                           )

process.TrackerGeometryWriter = cms.EDAnalyzer("PGeometricDetBuilder")
process.TrackerGeometryExtraWriter = cms.EDAnalyzer("PGeometricDetExtraBuilder")
process.TrackerParametersWriter = cms.EDAnalyzer("PTrackerParametersDBBuilder")

process.CondDB.timetype = cms.untracked.string('runnumber')
process.CondDB.connect = cms.string('sqlite_file:myfilerun3.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('IdealGeometryRecord'),tag = cms.string('TKRECO_Geometry_Run3_Test02')),
                                                            cms.PSet(record = cms.string('PGeometricDetExtraRcd'),tag = cms.string('TKExtra_Geometry_Run3_Test02')),
                                                            cms.PSet(record = cms.string('PTrackerParametersRcd'),tag = cms.string('TKParameters_Geometry_Run3_Test02'))
                                                            )
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.TrackerGeometryWriter*process.TrackerGeometryExtraWriter*process.TrackerParametersWriter)

