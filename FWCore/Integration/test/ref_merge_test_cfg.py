import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("file:ref_merge.root")
                            )

process.tester = cms.EDAnalyzer("OtherThingAnalyzer",
                                other = cms.untracked.InputTag("d","testUserTag"))

process.e = cms.EndPath(process.tester)

