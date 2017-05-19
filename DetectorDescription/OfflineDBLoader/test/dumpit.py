import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.maxEvents = cms.untracked.PSet(
            input = cms.untracked.int32(1)
                    )
process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )
process.load = cms.EDAnalyzer("OutputDDToDDL",
                            rotNumSeed = cms.int32(0),
                            fileName = cms.untracked.string("fred.xml")
                            )

process.myprint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.load)
process.ep = cms.EndPath(process.myprint)
