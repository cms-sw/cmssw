import FWCore.ParameterSet.Config as cms

process = cms.Process("testevtloop")
process.load("FWCore.MessageService.MessageLogger_cfi")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
process.maxLuminosityBlocks=cms.untracked.PSet(
    input=cms.untracked.int32(-1)
)

#process.source = cms.Source("EmptySource",
#     numberEventsInRun = cms.untracked.uint32(45),
#     firstRun = cms.untracked.uint32(122314),
#     numberEventsInLuminosityBlock = cms.untracked.uint32(1),
#     firstLuminosityBlock = cms.untracked.uint32(1)
#)

process.source= cms.Source("PoolSource",
              processingMode=cms.untracked.string('RunsAndLumis'),
              fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/PromptSkimCommissioning_v1/000/122/314/10D7BE65-3FD9-DE11-BED4-0026189438F4.root'),             
              firstRun=cms.untracked.uint32(122314),
              firstLuminosityBlock = cms.untracked.uint32(1),        
              firstEvent=cms.untracked.uint32(1),
              numberEventsInLuminosityBlock=cms.untracked.uint32(1)
             )
process.test = cms.EDAnalyzer("testEvtLoop")

process.p1 = cms.Path( process.test)

