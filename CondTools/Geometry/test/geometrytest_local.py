import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.autoCond import autoCond

process = cms.Process("GeometryTest")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(3),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("PCastorRcd"),
             tag = cms.string("CASTORRECO_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string("PZdcRcd"),
             tag = cms.string("ZDCRECO_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string("PCaloTowerRcd"),
             tag = cms.string("CTRECO_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string("PEcalEndcapRcd"),
             tag = cms.string("EERECO_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string("CSCRecoDigiParametersRcd"),
             tag = cms.string("CSCRECODIGI_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string("CSCRecoGeometryRcd"),
             tag = cms.string("CSCRECO_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string("PEcalBarrelRcd"),
             tag = cms.string("EBRECO_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string("RPCRecoGeometryRcd"),
             tag = cms.string("RPCRECO_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string("DTRecoGeometryRcd"),
             tag = cms.string("DTRECO_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string("PEcalPreshowerRcd"),
             tag = cms.string("EPRECO_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string("GeometryFileRcd"),
             tag = cms.string("XMLFILE_Geometry_Extended_TagXX"),
             connect = cms.string("sqlite_file:myfile.db"),
#             label = cms.string("Extended")
             ),
    cms.PSet(record = cms.string("IdealGeometryRecord"),
             tag = cms.string("TKRECO_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string('PTrackerParametersRcd'),
             tag = cms.string('TKParameters_Geometry_TagXX'),
             connect = cms.string("sqlite_file:myfile.db")
             ),
    cms.PSet(record = cms.string("PHcalRcd"),
             tag = cms.string("HCALRECO_Geometry_TagXX"),
             connect = cms.string("sqlite_file:myfile.db")
             )
    )

process.MessageLogger = cms.Service("MessageLogger")
process.demo = cms.EDAnalyzer("PrintEventSetupContent")

process.GeometryTester = cms.EDAnalyzer("GeometryTester",
                                        XMLTest = cms.untracked.bool(True),
                                        TrackerTest = cms.untracked.bool(True),
                                        EcalTest = cms.untracked.bool(True),
                                        HcalTest = cms.untracked.bool(True),
                                        HGCalTest = cms.untracked.bool(False),
                                        CaloTowerTest = cms.untracked.bool(True),
                                        CastorTest = cms.untracked.bool(True),
                                        ZDCTest = cms.untracked.bool(True),
                                        CSCTest = cms.untracked.bool(True),
                                        DTTest = cms.untracked.bool(True),
                                        RPCTest = cms.untracked.bool(True),
                                        geomLabel = cms.untracked.string("")
                                        )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.GeometryTester) #replace this with process.demo if you want to see the PrintEventSetupContent output.

