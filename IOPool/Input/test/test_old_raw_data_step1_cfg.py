import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring("file:"+sys.argv[1])
)

process.getTriggerNames = cms.EDAnalyzer("edmtest::GetTriggerNamesAnalyzer")

process.out = cms.OutputModule("PoolOutputModule",
   fastCloning = cms.untracked.bool(False),
   fileName = cms.untracked.string('file:converted.root')
)

process.endpath1 = cms.EndPath(process.getTriggerNames+process.out) 
