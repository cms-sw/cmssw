import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpMuonGeometryTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml'),
                                            appendToDataLabel = cms.string('MUON')
                                            )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  DDDetector = cms.ESInputTag('','MUON')
                                  )

process.p = cms.Path(process.testDump)
