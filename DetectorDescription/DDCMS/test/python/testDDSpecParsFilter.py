import FWCore.ParameterSet.Config as cms

process = cms.Process("DDSpecParsTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml'),
                                            appendToDataLabel = cms.string('MUON')
                                            )

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     appendToDataLabel = cms.string('MUON')
                                                     )

process.test = cms.EDAnalyzer("DDTestSpecParsFilter",
                              DDDetector = cms.ESInputTag('MUON'),
                              attribute = cms.untracked.string('MuStructure'),
                              value = cms.untracked.string('MuonBarrelDT')
                              )

process.p = cms.Path(process.test)
