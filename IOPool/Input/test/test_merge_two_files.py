import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("MERGETWOFILES")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("file:"+sys.argv[1],
                                                              "file:"+sys.argv[2]),
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck")
)

process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

process.out = cms.OutputModule("PoolOutputModule",
                            fileName = cms.untracked.string("merged_files.root"))

process.p = cms.Path(process.thingWithMergeProducer)

process.t = cms.EndPath(process.out)
