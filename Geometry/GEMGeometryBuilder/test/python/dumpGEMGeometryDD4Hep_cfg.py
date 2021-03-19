import FWCore.ParameterSet.Config as cms

process = cms.Process('DUMP')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load("Geometry.GEMGeometryBuilder.gemGeometry_cff")
process.load("Geometry.GEMGeometryBuilder.gemGeometryDump_cfi")

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/MuonCommonData/data/dd4hep/testGE0XML.xml'),
                                            appendToDataLabel = cms.string('')
                                        )

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                 appendToDataLabel = cms.string('')
)

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.Geometry=dict()
    process.MessageLogger.GEMNumberingScheme=dict()
    process.MessageLogger.GEMGeometry=dict()

process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.muonGeometryConstants.fromDD4Hep = True
process.gemGeometry.fromDDD = False
process.gemGeometry.fromDD4Hep = True
#process.gemGeometryDump.verbose = True

process.p = cms.Path(process.gemGeometryDump)
