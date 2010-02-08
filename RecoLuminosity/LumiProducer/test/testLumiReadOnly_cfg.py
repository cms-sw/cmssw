import FWCore.ParameterSet.Config as cms

process = cms.Process("readlumiback")
process.load("FWCore.MessageService.MessageLogger_cfi")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
#process.maxLuminosityBlocks=cms.untracked.PSet(
#    input=cms.untracked.int32(3)
#)
#process.maxEvents = cms.untracked.PSet(
#  input = cms.untracked.int32(3)
#)
process.source= cms.Source("PoolSource",
              fileNames=cms.untracked.vstring('file:testLumiProd.root'),
              firstRun=cms.untracked.uint32(1119985),
              firstLuminosityBlock = cms.untracked.uint32(1),                           
              firstEvent=cms.untracked.uint32(1),
             )
process.test = cms.EDAnalyzer("TestLumiProducer")

process.p1 = cms.Path( process.test )

