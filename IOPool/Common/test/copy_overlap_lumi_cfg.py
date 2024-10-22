import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("file:test_overlap_lumi.root"),
                            noRunLumiSort = cms.untracked.bool(True) )

process.out = cms.OutputModule("PoolOutputModule", fileName = cms.untracked.string("copy_test_overlap_lumi.root"))

process.o = cms.EndPath(process.out)
