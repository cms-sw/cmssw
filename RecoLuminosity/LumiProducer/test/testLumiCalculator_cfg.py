import FWCore.ParameterSet.Config as cms

process = cms.Process("LumiCalculator")
process.load("FWCore.MessageService.MessageLogger_cfi")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
process.maxLuminosityBlocks=cms.untracked.PSet(
    input=cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
   suppressInfo = cms.untracked.vstring(),
   destinations = cms.untracked.vstring('lumioutput'),
   categories = cms.untracked.vstring('LumiReport'),
   lumioutput = cms.untracked.PSet(
     threshold = cms.untracked.string('INFO'),
     noLineBreaks = cms.untracked.bool(True),
     noTimeStamps = cms.untracked.bool(True),
     INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
     LumiReport = cms.untracked.PSet( limit = cms.untracked.int32(10000000) )
   )
)
process.source= cms.Source("PoolSource",
              processingMode=cms.untracked.string('RunsAndLumis'),          
              fileNames=cms.untracked.vstring('rfio:/castor/cern.ch/user/x/xiezhen/MinBiasPromptSkimProcessed-122314.root'),
              firstRun=cms.untracked.uint32(122314),
              firstLuminosityBlock = cms.untracked.uint32(1),
              firstEvent=cms.untracked.uint32(1)
             )
process.test = cms.EDAnalyzer("LumiCalculator",
              showTriggerInfo= cms.untracked.bool(False)
             )

process.p1 = cms.Path( process.test )

