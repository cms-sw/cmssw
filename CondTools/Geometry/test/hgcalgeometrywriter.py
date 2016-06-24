import FWCore.ParameterSet.Config as cms

process = cms.Process("HGCalGeometryWriter")
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.Geometry.GeometryExtended2023sim_cff')

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.HGCalEEParametersWriter = cms.EDAnalyzer("PHGCalParametersDBBuilder",
                                                 Name = cms.untracked.string("HGCalEESensitive"),
                                                 NameW = cms.untracked.string("HGCalWafer"),
                                                 NameC = cms.untracked.string("HGCalCell"))

process.HGCalHEParametersWriter = cms.EDAnalyzer("PHGCalParametersDBBuilder",
                                                 Name = cms.untracked.string("HGCalHESiliconSensitive"),
                                                 NameW = cms.untracked.string("HGCalWafer"),
                                                 NameC = cms.untracked.string("HGCalCell"))

process.HGCalHEScParametersWriter = cms.EDAnalyzer("PHGCalParametersDBBuilder",
                                                   Name = cms.untracked.string("HGCalHEScintillatorSensitive"),
                                                   NameW = cms.untracked.string("HGCalWafer"),
                                                   NameC = cms.untracked.string("HGCalCell"))

process.CondDB.timetype = cms.untracked.string('runnumber')
process.CondDB.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(
                                                            cms.PSet(record = cms.string('PHGCalParametersRcd'),tag = cms.string('HGCALParameters_Geometry_Test01'))
                                                            )
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.HGCalEEParametersWriter)
