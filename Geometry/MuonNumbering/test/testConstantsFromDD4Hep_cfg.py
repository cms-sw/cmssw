import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonConstantsTest")
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Geometry.CMSCommonData.cmsExtendedGeometry2026D46XML_cfi")
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2026D46.xml'),
                                            appendToDataLabel = cms.string('')
                                            )

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                 appendToDataLabel = cms.string('')
)

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.muonGeometryConstants.fromDD4Hep = True

process.test = cms.EDAnalyzer("MuonGeometryConstantsTester")

process.p1 = cms.Path(process.test)
