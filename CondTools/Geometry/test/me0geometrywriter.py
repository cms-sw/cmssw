import FWCore.ParameterSet.Config as cms

process = cms.Process("ME0GeometryWriter")
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.Geometry.GeometryExtended2023sim_cff')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.ME0GeometryWriter = cms.EDAnalyzer("ME0RecoIdealDBLoader")

process.CondDB.timetype = cms.untracked.string('runnumber')
process.CondDB.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('ME0RecoGeometryRcd'),tag = cms.string('ME0RECO_Geometry_Test01'))
                                                            )
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.ME0GeometryWriter)
