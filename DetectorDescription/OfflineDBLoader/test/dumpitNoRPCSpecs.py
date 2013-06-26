import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.load("DetectorDescription.OfflineDBLoader.test.cmsIdealGeometryNoRPCSpecs_cfi")

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(1)
                    )
process.source = cms.Source("EmptyIOVSource",
#                            lastRun = cms.untracked.uint32(1),
#                            timetype = cms.string('runnumber'),
#                            firstRun = cms.untracked.uint32(1),
#                            interval = cms.uint32(1)
#                                                        )
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )
process.load = cms.EDAnalyzer("OutputDDToDDL",
                            rotNumSeed = cms.int32(0),
                            fileName = cms.untracked.string("fredNoRPCSpecs.xml")
                            )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.load)
process.ep = cms.EndPath(process.myprint)
