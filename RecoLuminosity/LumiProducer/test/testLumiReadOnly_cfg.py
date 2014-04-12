import FWCore.ParameterSet.Config as cms

process = cms.Process("readlumiback")
process.load("FWCore.MessageService.MessageLogger_cfi")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
process.maxLuminosityBlocks=cms.untracked.PSet(
    input=cms.untracked.int32(-1)
)
#process.maxEvents = cms.untracked.PSet(
#  input = cms.untracked.int32(3)
#)
process.source= cms.Source("PoolSource",
              fileNames=cms.untracked.vstring('file:testLumiProd-149181.root'),
              firstRun=cms.untracked.uint32(149181)
             )
process.test = cms.EDAnalyzer("TestLumiProducer")

process.p1 = cms.Path( process.test )

