import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("TEST")

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring("file:"+sys.argv[2])
)

process.out = cms.OutputModule("PoolOutputModule",
   fastCloning = cms.untracked.bool(False),
   fileName = cms.untracked.string('file:converted.root')
)

process.endpath1 = cms.EndPath(process.out) 
