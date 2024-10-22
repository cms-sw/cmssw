import FWCore.ParameterSet.Config as cms

process = cms.Process("dbtest")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
#  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

_messageSettings = cms.untracked.PSet(
                reportEvery = cms.untracked.int32(1),
                            limit = cms.untracked.int32(10000000)
                        )
process.MessageLogger.cerr.GetManyWithoutRegistration = _messageSettings
process.MessageLogger.cerr.GetByLabelWithoutRegistration = _messageSettings

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource",
     numberEventsInRun = cms.untracked.uint32(10),
     firstRun = cms.untracked.uint32(124020),
     numberEventsInLuminosityBlock = cms.untracked.uint32(1),
     firstLuminosityBlock = cms.untracked.uint32(1)
)

process.DBService=cms.Service("DBService",
           authPath=cms.untracked.string('/afs/cern.ch/cms/DB/lumi')
)
process.lumiProducer=cms.EDProducer("LumiProducer",
#           connect=cms.string('oracle://cms_orcoff_prep/cms_lumi_dev_offline'),
            connect=cms.string('frontier://cmsfrontier.cern.ch:8000/LumiPrep/CMS_LUMI_DEV_OFFLINE'),
#           connect=cms.string('frontier://LumiPrep/CMS_LUMI_DEV_OFFLINE'),                         
#           siteconfpath=cms.untracked.string('/afs/cern.ch/user/x/xiezhen/w1/lumical/CMSSW_3_5_0_pre5/src/RecoLuminosity/LumiProducer'),
           lumiversion=cms.untracked.string('0001') 
)
process.test = cms.EDAnalyzer("TestLumiProducer")
process.out = cms.OutputModule("PoolOutputModule",
           fileName=cms.untracked.string("testLumiProd.root")
)
process.p1 = cms.Path(process.lumiProducer * process.test)
process.e = cms.EndPath(process.out)
