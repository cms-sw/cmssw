import FWCore.ParameterSet.Config as cms

process = cms.Process("getLumiRaw")
process.load("FWCore.MessageService.MessageLogger_cfi")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
#
#keep maxEvents equal to numberEventsInRun so that we augment LS
#
process.maxEvents=cms.untracked.PSet(
    input=cms.untracked.int32(100)
)
process.maxLuminosityBlocks=cms.untracked.PSet(
    input=cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource",                            
     numberEventsInRun = cms.untracked.uint32(100),
     firstRun = cms.untracked.uint32(122314),
     numberEventsInLuminosityBlock = cms.untracked.uint32(10),
     firstLuminosityBlock = cms.untracked.uint32(1)
)

process.genlumiraw = cms.EDAnalyzer("genLumiRaw")

process.p1 = cms.Path( process.genlumiraw )

