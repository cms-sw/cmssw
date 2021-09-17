import FWCore.ParameterSet.Config as cms

DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                    confGeomXMLFiles = cms.FileInPath(''),
                                    appendToDataLabel = cms.string('')
)

DDSpecParRegistryESProducer = cms.ESProducer("DDSpecParRegistryESProducer",
                                             appendToDataLabel = cms.string('')
)

DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                            appendToDataLabel = cms.string(''))

DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                         appendToDataLabel = cms.string('')
)
