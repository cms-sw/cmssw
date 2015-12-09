import FWCore.ParameterSet.Config as cms

process = cms.Process("CaloGeometryWriter")
process.load('CondCore.DBCommon.CondDBCommon_cfi')
process.load('Configuration.Geometry.GeometryExtended2023_cff')
process.load('Geometry.CaloEventSetup.CaloGeometry2023DBWriter_cfi')
process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.CaloGeometryWriter = cms.EDAnalyzer("PCaloGeometryBuilder",
                                            EcalE = cms.untracked.bool(False),
                                            EcalP = cms.untracked.bool(False),
                                            HGCal = cms.untracked.bool(True))

process.HcalParametersWriter = cms.EDAnalyzer("HcalParametersDBBuilder")
##process.prod = cms.EDAnalyzer("HGCalGeometryTester")

process.CondDBCommon.BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
process.CondDBCommon.timetype = cms.untracked.string('runnumber')
process.CondDBCommon.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('PEcalBarrelRcd'),   tag = cms.string('EBRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PHGCalRcd'),         tag = cms.string('HGCALRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PHcalRcd'),         tag = cms.string('HCALRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('HcalParametersRcd'), tag = cms.string('HCALParameters_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PCaloTowerRcd'),    tag = cms.string('CTRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PZdcRcd'),          tag = cms.string('ZDCRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PCastorRcd'),       tag = cms.string('CASTORRECO_Geometry_Test01'))
                                                            )
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

##process.p1 = cms.Path(process.prod)
process.p1 = cms.Path(process.CaloGeometryWriter*process.HcalParametersWriter)
