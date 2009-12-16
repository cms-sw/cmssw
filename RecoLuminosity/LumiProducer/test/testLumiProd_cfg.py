
import FWCore.ParameterSet.Config as cms

process = cms.Process("standalonetest")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("RecoLuminosity.LumiProducer.nonGlobalTagLumiProducerPrep_cff")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
#  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.maxLuminosityBlocks=cms.untracked.PSet(
    input=cms.untracked.int32(-1)
)


#process.source = cms.Source("EmptySource",
#     numberEventsInRun = cms.untracked.uint32(21),
#     firstRun = cms.untracked.uint32(83037),
#     numberEventsInLuminosityBlock = cms.untracked.uint32(1),
#     firstLuminosityBlock = cms.untracked.uint32(1)
#)

#process.source = cms.Source("EmptyIOVSource",
#    timetype = cms.string('lumiid'),
#    firstValue = cms.uint64(515481974865922),
#    lastValue = cms.uint64(515481974866107),
#    interval = cms.uint64(1)
#)

process.source= cms.Source("PoolSource",
             processingMode=cms.untracked.string('RunsAndLumis'),        
             #fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/PromptSkimCommissioning_v1/000/122/314/10D7BE65-3FD9-DE11-BED4-0026189438F4.root'),
             fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/PromptSkimCommissioning_v1/000/124/025/3CBFB201-B5E7-DE11-8648-0026189438E8.root'),
             firstRun=cms.untracked.uint32(124025),
             firstLuminosityBlock = cms.untracked.uint32(1),                           
             firstEvent=cms.untracked.uint32(1),
             numberEventsInLuminosityBlock=cms.untracked.uint32(1)
             )

process.LumiESSource.DBParameters.authenticationPath=cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
process.LumiESSource.BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
process.LumiESSource.connect=cms.string('sqlite_file:/afs/cern.ch/user/x/xiezhen/w1/offlinelumi.db')
process.LumiESSource.toGet=cms.VPSet(
    cms.PSet(
      record = cms.string('LumiSectionDataRcd'),
      tag = cms.string('collision')
    )
)

process.lumiProducer=cms.EDProducer("LumiProducer")
process.test = cms.EDAnalyzer("TestLumiProducer")

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('MinBiasPromptSkimProcessed-124025.root')
)

process.p1 = cms.Path(process.lumiProducer * process.test)

process.e = cms.EndPath(process.out)
