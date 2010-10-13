import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("OLDREAD")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("file:"+sys.argv[2]))

process.tester = cms.EDAnalyzer("TestFindProduct",
                                inputTags = cms.untracked.VInputTag(cms.InputTag("i")),
                                expectedSum = cms.untracked.int32(40))

process.out = cms.OutputModule("PoolOutputModule",
                            fileName = cms.untracked.string("old.root"))

process.t = cms.EndPath(process.tester*process.out)
