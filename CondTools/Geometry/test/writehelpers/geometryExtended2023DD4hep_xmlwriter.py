import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryXMLWriter")

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2023.xml'),
                                            appendToDataLabel = cms.string('make-payload')
                                           )

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                  appendToDataLabel = cms.string('make-payload')
                                                )

process.BigXMLWriter = cms.EDAnalyzer("OutputDD4hepToDDL",
                              fileName = cms.untracked.string("./geSingleBigFile.xml")
                              )


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.BigXMLWriter)

