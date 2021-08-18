import FWCore.ParameterSet.Config as cms

process = cms.Process("DDCMSDetectorTest")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['upgrade2021']

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            rootDDName = cms.string('cms:OCMS'),
                                            label = cms.string('Extended'),
                                            fromDB = cms.bool(True),
                                            appendToDataLabel = cms.string('CMS')
)

process.DDVectorRegistryESProducer = cms.ESProducer("DDVectorRegistryESProducer",
                                                    appendToDataLabel = cms.string('CMS')
                                                    )

process.test = cms.EDAnalyzer("DDCMSDetector",
                              DDDetector = cms.ESInputTag('','CMS')
                              )

process.testVectors = cms.EDAnalyzer("DDTestVectors",
                                     DDDetector = cms.ESInputTag('','CMS')
                                     )

process.testDump = cms.EDAnalyzer("DDTestDumpFile",
                                  DDDetector = cms.ESInputTag('','CMS')
                              )

process.testGeoIter = cms.EDAnalyzer("DDTestDumpGeometry",
                                     DDDetector = cms.ESInputTag('','CMS')
                                     )

process.p = cms.Path(
    process.test
    +process.testVectors
    +process.testDump
    +process.testGeoIter)
