import FWCore.ParameterSet.Config as cms

process = cms.Process("dbprodtest")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
#  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(10)
)
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.consumerName = 'DQM Luminosity Consumer'
#process.EventStreamHttpReader.sourceURL = cms.string('http://dqm-c2d07-30:50082/urn:xdaq-application:lid=29')
process.EventStreamHttpReader.SelectHLTOutput = cms.untracked.string('hltOutputALCALUMIPIXELS')
#process.source= cms.Source("PoolSource",
#           processingMode=cms.untracked.string('RunsAndLumis'),        
#           fileNames=cms.untracked.vstring(
#'file:/data/cmsdata/009F3522-D604-E111-A08D-003048F1183E.root')
#)

process.DBService=cms.Service("DBService",
           authPath=cms.untracked.string('/nfshome0/popcondev/conddb')
)
process.expressLumiProducer=cms.EDProducer("ExpressLumiProducer",
   connect=cms.string('oracle://cms_omds_lb/CMS_RUNTIME_LOGGER'),
)
process.test = cms.EDAnalyzer("TestExpressLumiProducer")
#
#uncomment the block if want to save to file
#process.out = cms.OutputModule("PoolOutputModule",
#  fileName = cms.untracked.string('testExpressLumiProd.root')
#)#
#process.e = cms.EndPath(process.out)

process.p1 = cms.Path(process.expressLumiProducer * process.test)

