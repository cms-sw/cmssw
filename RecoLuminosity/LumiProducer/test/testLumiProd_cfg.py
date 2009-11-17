
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

#process.maxEvents = cms.untracked.PSet(
#  input = cms.untracked.int32(10)
#)

#process.source = cms.Source("EmptySource",
#     numberEventsInRun = cms.untracked.uint32(21),
#     firstRun = cms.untracked.uint32(83037),
#     numberEventsInLuminosityBlock = cms.untracked.uint32(1),
#     firstLuminosityBlock = cms.untracked.uint32(1)
#)

#process.maxLuminosityBlocks=cms.untracked.PSet(
#    input=cms.untracked.int32(3)
#)

#process.source = cms.Source("EmptyIOVSource",
#    timetype = cms.string('lumiid'),
#    firstValue = cms.uint64(515481974865922),
#    lastValue = cms.uint64(515481974866107),
#    interval = cms.uint64(1)
#)

process.source= cms.Source("PoolSource",
             fileNames=cms.untracked.vstring('/store/relval/CMSSW_3_4_0_pre2/RelValSingleElectronPt10/GEN-SIM-RECO/MC_3XY_V10-v1/0003/BE702AE8-C0BD-DE11-87CA-002618943861.root')
#              firstRun=cms.untracked.uint32(120020),
#              firstLuminosityBlock = cms.untracked.uint32(1),                           
#              firstEvent=cms.untracked.uint32(1),
             )

process.LumiESSource.DBParameters.authenticationPath=cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
process.LumiESSource.BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
process.LumiESSource.connect=cms.string('sqlite_file:offlinelumi.db')
process.LumiESSource.toGet=cms.VPSet(
    cms.PSet(
      record = cms.string('LumiSectionDataRcd'),
      tag = cms.string('testlumiroot')
    )
)

process.lumiProducer=cms.EDProducer("LumiProducer")
process.test = cms.EDAnalyzer("TestLumiProducer")

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('MC3XYProcessed.root')
)

process.p1 = cms.Path(process.lumiProducer * process.test)

process.e = cms.EndPath(process.out)
