import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.fred = cms.ESSource("XMLIdealGeometryESSource"
                    ,geomXMLFiles = cms.vstring('GeometryReaders/XMLIdealGeometryESSource/test/fred.xml')
                    ,rootNodeName = cms.string('cms:OCMS')
                    )

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.demo = cms.EDAnalyzer("PrintEventSetupContent")

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.pAStd = cms.EDAnalyzer("PerfectGeometryAnalyzer"
                               ,ddRootNodeName = cms.string("cms:OCMS")
                               ,dumpPosInfo = cms.untracked.bool(True)
                               ,label = cms.untracked.string("")
                               ,isMagField = cms.untracked.bool(False)
                               ,dumpSpecs = cms.untracked.bool(True)
                               ,dumpGeoHistory = cms.untracked.bool(True)
                               ,outFileName = cms.untracked.string("STD")
                               ,numNodesToDump = cms.untracked.uint32(0)
                               )

#es_prefer_fred = cms.ESPrefer("XMLIdealGeometryESSource","fred")

process.pABF = cms.EDAnalyzer("PerfectGeometryAnalyzer"
                              ,ddRootNodeName = cms.string("cms:OCMS")
                              ,dumpPosInfo = cms.untracked.bool(True)
                              ,label = cms.untracked.string("fred")
                              ,isMagField = cms.untracked.bool(False)
                              ,dumpSpecs = cms.untracked.bool(True)
                              ,dumpGeoHistory = cms.untracked.bool(True)
                              ,outFileName = cms.untracked.string("BF")
                              ,numNodesToDump = cms.untracked.uint32(0)
                              )

process.BigXMLWriter = cms.EDAnalyzer("OutputDDToDDL",
                              rotNumSeed = cms.int32(0),
                              fileName = cms.untracked.string("fred.xml")
                              )

process.comparedddump = cms.EDAnalyzer("TestCompareDDDumpFiles"
                                       ,dumpFile1 = cms.string("dumpSTD")
                                       ,dumpFile2 = cms.string("dumpBF")
#                                       ,tolerance = cms.double(0.0004)
                                       )

process.Timing = cms.Service("Timing")

process.p1 = cms.Path(process.BigXMLWriter+process.pAStd+process.demo+process.pABF+process.demo+process.comparedddump)
#process.p1 = cms.Path(process.demo)
process.e1 = cms.EndPath(process.myprint)
