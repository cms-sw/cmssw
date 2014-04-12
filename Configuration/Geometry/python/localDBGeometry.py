import FWCore.ParameterSet.Config as cms

def customiseDBRecords(process):

    process.PoolDBESSourceGeometry = cms.ESSource("PoolDBESSource",
                                                  process.CondDBSetup,
                                                  timetype = cms.string('runnumber'),
                                                  toGet = cms.VPSet(
        #cms.PSet(record = cms.string('GeometryFileRcd'),         tag = cms.string('XMLFILE_Geometry_Extended_61YV5_mc')),
        cms.PSet(record = cms.string('PEcalBarrelRcd'),          tag = cms.string('EBRECO_Geometry_61YV5')),
        cms.PSet(record = cms.string('PEcalEndcapRcd'),          tag = cms.string('EERECO_Geometry_61YV5')),
        cms.PSet(record = cms.string('PEcalPreshowerRcd'),       tag = cms.string('EPRECO_Geometry_61YV5')),
        cms.PSet(record = cms.string('PHcalRcd'),                tag = cms.string('HCALRECO_Geometry_61YV5')),
        cms.PSet(record = cms.string('PCaloTowerRcd'),           tag = cms.string('CTRECO_Geometry_61YV5')),
        cms.PSet(record = cms.string('PZdcRcd'),                 tag = cms.string('ZDCRECO_Geometry_61YV5')),
        cms.PSet(record = cms.string('PCastorRcd'),              tag = cms.string('CASTORRECO_Geometry_61YV5'))
##                                                                 cms.PSet(record = cms.string('CSCRecoGeometryRcd'),      tag = cms.string('CSCRECO_Geometry_61YV5')),
##                                                                 cms.PSet(record = cms.string('CSCRecoDigiParametersRcd'),tag = cms.string('CSCRECODIGI_Geometry_61YV5')),
##                                                                 cms.PSet(record = cms.string('DTRecoGeometryRcd'),       tag = cms.string('DTRECO_Geometry_61YV5')),
##                                                                 cms.PSet(record = cms.string('IdealGeometryRecord'),     tag = cms.string('TKRECO_Geometry_61YV5')),
##                                                                 cms.PSet(record = cms.string('PGeometricDetExtraRcd'),   tag = cms.string('TKExtra_Geometry_61YV5')),
##                                                                 cms.PSet(record = cms.string('PZdcRcd'),                 tag = cms.string('ZDCRECO_Geometry_61YV5')),
##                                                                 cms.PSet(record = cms.string('RPCRecoGeometryRcd'),      tag = cms.string('RPCRECO_Geometry_61YV5'))
        ),
                                                  connect = cms.string('sqlite_file:myfile.db')
                                                  )

    process.es_prefer_geometry = cms.ESPrefer( "PoolDBESSource", "PoolDBESSourceGeometry" )

    return process
