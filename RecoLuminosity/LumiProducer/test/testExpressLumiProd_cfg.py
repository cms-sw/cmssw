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
process.source= cms.Source("PoolSource",
           processingMode=cms.untracked.string('RunsAndLumis'),        
           fileNames=cms.untracked.vstring(
'file:/data/cmsdata/009F3522-D604-E111-A08D-003048F1183E.root')
)
process.DBService=cms.Service("DBService",
           authPath=cms.untracked.string('/data/cmsdata')
)
process.expressLumiProducer=cms.EDProducer("ExpressLumiProducer",
   connect=cms.string('oracle://cms_orcoff_prod/cms_runtime_logger'),
)
process.test = cms.EDAnalyzer("TestExpressLumiProducer")

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('testExpressLumiProd.root')
)
process.p1 = cms.Path(process.expressLumiProducer * process.test)
process.e = cms.EndPath(process.out)
