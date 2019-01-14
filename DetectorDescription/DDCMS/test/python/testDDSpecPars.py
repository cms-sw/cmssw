import FWCore.ParameterSet.Config as cms

process = cms.Process("DDSpecParsTest")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('DetectorDescription/DDCMS/data/cms-2015-muon-geometry.xml'),
                                            label = cms.string('MUON')
                                            )

process.DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                                     label = cms.string('MUON')
                                                     )

process.test = cms.EDAnalyzer("DDTestSpecPars",
                              fromDataLabel = cms.untracked.string('MUON')
                              )

process.p = cms.Path(process.test)
