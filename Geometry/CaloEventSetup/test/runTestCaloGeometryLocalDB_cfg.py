
import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load('Configuration/StandardSequences/GeometryDB_cff')

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_3XY_V10::All'

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.PoolDBESSourceGeometry = cms.ESSource("PoolDBESSource",
                                              process.CondDBSetup,
                                              timetype = cms.string('runnumber'),
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
                                                                ),
                                              connect = cms.string('sqlite_file:myfile.db')
                                              )

process.es_prefer_geometry = cms.ESPrefer( "PoolDBESSource", "PoolDBESSourceGeometry" )


process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(4) )

process.source = cms.Source("EmptySource")

process.etta = cms.EDAnalyzer("dumpEcalTrigTowerMapping")

process.ctgw = cms.EDAnalyzer("testEcalGetWindow")

process.cga = cms.EDAnalyzer("CaloGeometryAnalyzer",
                             fullEcalDump = cms.untracked.bool(True)
                             )

process.mfa = cms.EDAnalyzer("testMagneticField")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('calogeom.root')
                                   )

process.p1 = cms.Path(process.etta*process.ctgw*process.cga*process.mfa)


