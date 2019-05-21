import FWCore.ParameterSet.Config as cms

process = cms.Process("DDVectorsTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            appendToDataLabel = cms.string('CMS'),
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml')
                                            )

process.DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                                    appendToDataLabel = cms.string('CMS'))

process.test = cms.EDAnalyzer("DDTestVectors",
                              DDDetector = cms.ESInputTag('','CMS')
                              )

process.p = cms.Path(process.test)
