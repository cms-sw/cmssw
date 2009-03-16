import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryWriter")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Geometry/CaloEventSetup/CaloGeometryDBWriter_cfi')


process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.BigXMLWriter = cms.EDAnalyzer("OutputDDToDDL",
                              rotNumSeed = cms.int32(0),
                              fileName = cms.untracked.string("fred.xml")
                              )

process.XMLGeometryWriter = cms.EDAnalyzer("XMLGeometryBuilder",
                                           XMLFileName = cms.untracked.string("./fred.xml"),
                                           ZIP = cms.untracked.bool(True)
                                           )
process.TrackerGeometryWriter = cms.EDAnalyzer("PGeometricDetBuilder")

process.CaloGeometryWriter = cms.EDAnalyzer("PCaloGeometryBuilder")

process.CSCGeometryWriter = cms.EDAnalyzer("CSCRecoIdealDBLoader")

process.DTGeometryWriter = cms.EDAnalyzer("DTRecoIdealDBLoader")

process.RPCGeometryWriter = cms.EDAnalyzer("RPCRecoIdealDBLoader")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),

                                          timetype = cms.untracked.string('runnumber'),
                                          connect = cms.string('sqlite_file:myfile.db'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'),tag = cms.string('XMLFILE_Geometry_Test03')),
                                                            cms.PSet(record = cms.string('IdealGeometryRecord'),tag = cms.string('TKRECO_Geometry_Test02')),
                                                            cms.PSet(record = cms.string('PEcalBarrelRcd'),   tag = cms.string('EBRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PEcalEndcapRcd'),   tag = cms.string('EERECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PEcalPreshowerRcd'),tag = cms.string('EPRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PHcalRcd'),         tag = cms.string('HCALRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PCaloTowerRcd'),    tag = cms.string('CTRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PZdcRcd'),          tag = cms.string('ZDCRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('PCastorRcd'),       tag = cms.string('CASTORRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('CSCRecoGeometryRcd'),tag = cms.string('CSCRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('CSCRecoDigiParametersRcd'),tag = cms.string('CSCRECODIGI_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('DTRecoGeometryRcd'),tag = cms.string('DTRECO_Geometry_Test01')),
                                                            cms.PSet(record = cms.string('RPCRecoGeometryRcd'),tag = cms.string('RPCRECO_Geometry_Test01'))
                                                            )
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.BigXMLWriter+process.XMLGeometryWriter+process.TrackerGeometryWriter+process.CaloGeometryWriter+process.CSCGeometryWriter+process.DTGeometryWriter+process.RPCGeometryWriter)

