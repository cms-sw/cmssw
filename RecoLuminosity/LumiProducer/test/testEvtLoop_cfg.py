import FWCore.ParameterSet.Config as cms

process = cms.Process("testevtloop")
process.load("FWCore.MessageService.MessageLogger_cfi")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
process.maxLuminosityBlocks=cms.untracked.PSet(
    input=cms.untracked.int32(44)
)

process.source = cms.Source("EmptySource",
     numberEventsInRun = cms.untracked.uint32(45),
     firstRun = cms.untracked.uint32(122314),
     numberEventsInLuminosityBlock = cms.untracked.uint32(1),
     firstLuminosityBlock = cms.untracked.uint32(1)
)

#process.source= cms.Source("PoolSource",
#              processingMode=cms.untracked.string('RunsAndLumis'),
#              fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/AlCaPhiSymEcal/ALCARECO/v2/000/122/314/D662B7EB-88D8-DE11-99D4-001D09F251BD.root'),
#              firstRun=cms.untracked.uint32(122314),
#              firstLuminosityBlock = cms.untracked.uint32(1),        
#              firstEvent=cms.untracked.uint32(1),
#              numberEventsInLuminosityBlock=cms.untracked.uint32(1)
#              #numberEventsInRun=cms.untracked.uint32(50),
#             )
process.test = cms.EDAnalyzer("testEvtLoop")

process.p1 = cms.Path( process.test)

