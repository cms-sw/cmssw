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
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        lumioutput = cms.untracked.PSet(
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            LumiReport = cms.untracked.PSet(
                limit = cms.untracked.int32(10000000)
            ),
            noLineBreaks = cms.untracked.bool(True),
            noTimeStamps = cms.untracked.bool(True),
            threshold = cms.untracked.string('INFO')
        )
    ),
    suppressInfo = cms.untracked.vstring()
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

