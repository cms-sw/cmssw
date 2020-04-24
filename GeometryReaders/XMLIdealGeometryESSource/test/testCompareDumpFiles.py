import FWCore.ParameterSet.Config as cms

process = cms.Process("CompareGeometryTest")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.MessageLogger = cms.Service("MessageLogger",
                                    compDDdumperrors = cms.untracked.PSet( threshold = cms.untracked.string('ERROR')),
                                    destinations = cms.untracked.vstring('compDDdumperrors')
                                    )

process.comparedddump = cms.EDAnalyzer("TestCompareDDDumpFiles",
                                       dumpFile1 = cms.string('workarea/xml/dumpSTD'),
                                       dumpFile2 = cms.string('workarea/db/dumpBDB'),
                                       tolerance = cms.untracked.double(0.0004)
                                       )

process.p1 = cms.Path(process.comparedddump)
process.e1 = cms.EndPath(process.myprint)
