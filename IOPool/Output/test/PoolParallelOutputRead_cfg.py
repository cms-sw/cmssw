import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("TESTOUTPUTREAD")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( ['file:'+x for x in os.listdir('.') if x[-5:]=='.root' and x[:-6]=='PoolOutputTest_' ] )
)



