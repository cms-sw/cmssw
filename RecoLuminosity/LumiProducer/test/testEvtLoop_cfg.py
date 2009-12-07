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
#process.maxEvents = cms.untracked.PSet(
#  input = cms.untracked.int32(44)
#)
process.source= cms.Source("PoolSource",
              processingMode=cms.untracked.string('RunsAndLumis'),
              #fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/AlCaPhiSymEcal/ALCARECO/v2/000/122/314/D662B7EB-88D8-DE11-99D4-001D09F251BD.root'),
              #fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/FEDMonitor/RAW/v1/000/122/314/10F83DFC-7CD8-DE11-917D-001D09F248F8.root'),
              #fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/AlCaPhiSymEcal/RAW/v1/000/122/314/BA589404-7DD8-DE11-848F-001D09F27067.root'),
              #fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/AlCaP0/ALCARECO/v2/000/123/596/F81B4571-8BE2-DE11-8E2D-003048D2C1C4.root','/store/data/BeamCommissioning09/AlCaP0/ALCARECO/v2/000/123/596/DE861A27-8CE2-DE11-8A09-003048D2C1C4.root'),
              fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/AlCaPhiSymEcal/ALCARECO/v2/000/123/596/FCEEEE81-8BE2-DE11-9334-001617DC1F70.root'),
              #firstRun=cms.untracked.uint32(122314),
              firstRun=cms.untracked.uint32(123596),             
              firstLuminosityBlock = cms.untracked.uint32(1),        
              firstEvent=cms.untracked.uint32(1),
              numberEventsInLuminosityBlock=cms.untracked.uint32(1)
              #numberEventsInRun=cms.untracked.uint32(50),
             )
process.test = cms.EDAnalyzer("testEvtLoop")

process.p1 = cms.Path( process.test)

