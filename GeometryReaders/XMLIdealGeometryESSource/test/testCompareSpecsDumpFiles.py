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
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        compDDdumpdiff = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO')
        ),
        compDDdumperrors = cms.untracked.PSet(
            threshold = cms.untracked.string('ERROR')
        )
    )
)

process.comparedddump = cms.EDAnalyzer("TestCompareDDSpecsDumpFiles",
                                       dumpFile1 = cms.string("diff/dumpSpecsdumpGTDB.sorted"),
                                       dumpFile2 = cms.string("diff/dumpSpecsdumpSTD.sorted"),
                                       tolerance = cms.untracked.double(0.0004)
                                       )

process.p1 = cms.Path(process.comparedddump)
process.e1 = cms.EndPath(process.myprint)
