
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

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

#process.source = cms.Source("EmptySource",
#     numberEventsInRun = cms.untracked.uint32(21),
#     firstRun = cms.untracked.uint32(83037),
#     numberEventsInLuminosityBlock = cms.untracked.uint32(1),
#     firstLuminosityBlock = cms.untracked.uint32(1)
#)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('lumiid'),
    firstValue = cms.uint64(356641199357953),
    lastValue = cms.uint64(356641199357973),
    interval = cms.uint64(1)
)

process.LumiESSource.DBParameters.authenticationPath=cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
process.LumiESSource.BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
process.LumiESSource.connect=cms.string('sqlite_file:offlinelumi.db')
process.LumiESSource.toGet=cms.VPSet(
    cms.PSet(
      record = cms.string('LuminosityInfoRcd'),
      tag = cms.string('globalrun')
    ),
    cms.PSet(
      record = cms.string('HLTScalerRcd'),
      tag = cms.string('globalrunhltscaler')
    )
)

process.lumiProducer=cms.EDProducer("LumiProducer")
process.test = cms.EDAnalyzer("TestLumiProducer")

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('testLumiProd.root')
)

process.p1 = cms.Path(process.lumiProducer * process.test)

process.e = cms.EndPath(process.out)
