import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryCompareDDCompactView")

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.GeometryFileDump = cms.EDAnalyzer("CompareDDCompactViews",
                                          XMLFileName1 = cms.untracked.string("GeometryExtended2017.81YV15.xml"),
                                          XMLFileName2 = cms.untracked.string("GeometryExtended2017.81YV16.xml"),
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.GeometryFileDump)
