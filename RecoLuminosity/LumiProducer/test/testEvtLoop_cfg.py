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
              #fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/PromptSkimCommissioning_v1/000/122/314/10D7BE65-3FD9-DE11-BED4-0026189438F4.root'),
              fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/E83AA00B-340D-DF11-8D18-0018F3D09690.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/DC853BD1-2F0D-DF11-B193-0030486790B0.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/A8C81FDD-300D-DF11-A884-001A92971AAA.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/8CDF61BC-2D0D-DF11-AE53-001A92810AD2.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/8C45D7B9-310D-DF11-ACB1-0018F3D096EE.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/70B0A9D1-2F0D-DF11-8906-003048678FF8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/56C373E3-320D-DF11-97ED-001A92971B36.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/54BCEB0F-3D0D-DF11-9A3C-002618943826.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/4C48BDE4-300D-DF11-9771-001A92810AD2.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/4C2D49DD-300D-DF11-B9D0-001A92810AE4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/F825ED95-250D-DF11-A065-001A928116BC.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/F4102499-220D-DF11-88E6-003048679012.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/D8CCD1AA-220D-DF11-9C9A-003048678FF2.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/B2A4827C-250D-DF11-A145-001A92971B7C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/AE38B1A7-220D-DF11-83B3-001A928116D4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/A0D3AB95-250D-DF11-83FD-001A92971B20.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/8843608A-240D-DF11-96B4-001A92971B80.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/86AA623D-230D-DF11-82F4-0018F3D0960C.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/7E2E8196-250D-DF11-A71F-001A92971B5E.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/76103E8D-240D-DF11-B1B7-0018F3D09612.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/56E9DB8B-240D-DF11-B941-001A928116C6.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/48E04E8A-240D-DF11-96E4-001A928116D4.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/3E481BC6-220D-DF11-BF25-001A92971B7E.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/3CFBC896-250D-DF11-80A0-001A92971BD8.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/34E1688D-240D-DF11-8F56-0018F3D0967E.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/346039E8-1A0D-DF11-80C1-0018F3D096B6.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/303D7E92-250D-DF11-B1F5-002618943950.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/2E5DFC90-2B0D-DF11-B1A8-001A928116E6.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/0E83998B-240D-DF11-B327-001A92810AF2.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/0C72F496-260D-DF11-A727-003048D15E24.root'),
              #firstRun=cms.untracked.uint32(122314),
              #firstLuminosityBlock = cms.untracked.uint32(1),        
              #firstEvent=cms.untracked.uint32(1)
             )
process.test = cms.EDAnalyzer("testEvtLoop")

process.p1 = cms.Path( process.test)

