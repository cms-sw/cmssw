import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('CondCore.CondDB.CondDB_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.load('FWCore.MessageLogger.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.RPCGeometry=dict()


process.CondDB.timetype = cms.untracked.string('runnumber')
process.CondDB.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBESSourceGeometry = cms.ESSource("PoolDBESSource",
                                      process.CondDB,
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'),tag = cms.string('XMLFILE_Geometry_Extended_TagXX')),
                                                        cms.PSet(record = cms.string('IdealGeometryRecord'),tag = cms.string('TKRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PEcalBarrelRcd'),   tag = cms.string('EBRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PEcalEndcapRcd'),   tag = cms.string('EERECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PEcalPreshowerRcd'),tag = cms.string('EPRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PHcalRcd'),         tag = cms.string('HCALRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PCaloTowerRcd'),    tag = cms.string('CTRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PZdcRcd'),          tag = cms.string('ZDCRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PCastorRcd'),       tag = cms.string('CASTORRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('CSCRecoGeometryRcd'),tag = cms.string('CSCRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('CSCRecoDigiParametersRcd'),tag = cms.string('CSCRECODIGI_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('DTRecoGeometryRcd'),tag = cms.string('DTRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('RPCRecoGeometryRcd'),tag = cms.string('RPCRECO_Geometry_TagXX'))
                                                        )
                                      )

process.es_prefer_geometry = cms.ESPrefer( "PoolDBESSource", "PoolDBESSourceGeometry" )

process.test1 = cms.EDAnalyzer("RPCGEO")
process.test2 = cms.EDAnalyzer("RPCGeometryAnalyzer")

process.p = cms.Path(process.test1+process.test2)

