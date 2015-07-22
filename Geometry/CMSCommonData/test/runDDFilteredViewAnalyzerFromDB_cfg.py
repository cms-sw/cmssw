import FWCore.ParameterSet.Config as cms

process = cms.Process("DDFilteredViewTest")
process.load("Configuration.Geometry.GeometryDB_cff")
process.XMLFromDBSource.label=''
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.PoolDBESSourceGeometry = cms.ESSource("PoolDBESSource",
                               process.CondDBSetup,
                               timetype = cms.string('runnumber'),
                               toGet = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'),tag = cms.string('XMLFILE_Geometry_75YV4_Extended_mc')),
                                                 cms.PSet(record = cms.string('IdealGeometryRecord'),tag = cms.string('TKRECO_Geometry_75YV4'))
                                                  ),
                               connect = cms.string('sqlite_file:myfile.db')
                               )
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.fva = cms.EDAnalyzer("DDFilteredViewAnalyzer",
                             attribute = cms.string("OnlyForHcalSimNumbering"),
                             value = cms.string("any"))

process.p1 = cms.Path(process.fva)

