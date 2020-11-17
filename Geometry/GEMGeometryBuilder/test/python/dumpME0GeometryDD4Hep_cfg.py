import FWCore.ParameterSet.Config as cms

process = cms.Process('DUMP')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.GEMGeometryBuilder.gemGeometry_cff")
process.load("Geometry.GEMGeometryBuilder.me0Geometry_cff")
process.load("Geometry.GEMGeometryBuilder.me0GeometryDump_cfi")

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/MuonCommonData/data/dd4hep/testGEMXML.xml'),
                                            appendToDataLabel = cms.string('')
                                        )

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                 appendToDataLabel = cms.string('')
)


    
    
    

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.muonGeometryConstants.fromDD4Hep = True
process.me0Geometry.fromDDD = False
process.me0Geometry.fromDD4hep = True
#process.me0GeometryDump.verbose = True

process.p = cms.Path(process.me0GeometryDump)
