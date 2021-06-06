import FWCore.ParameterSet.Config as cms


_messageSettings = cms.untracked.PSet(
                reportEvery = cms.untracked.int32(1),
                            limit = cms.untracked.int32(10000000)
                        )
process.MessageLogger.cerr.GetManyWithoutRegistration = _messageSettings
process.MessageLogger.cerr.GetByLabelWithoutRegistration = _messageSettings

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
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/00E988DE-095B-DF11-B111-001D09F2A465.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/00AFE4C3-395B-DF11-969C-0030486733D8.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/00AC9715-5B5B-DF11-B18A-003048D37538.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/00948B8C-305B-DF11-B879-0030487CD906.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/0084A8A0-375B-DF11-9616-0019B9F730D2.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/006996D6-4C5B-DF11-BC60-000423D98750.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/005EEFFC-1E5B-DF11-B014-000423D99AA2.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/00538ED8-6D5B-DF11-8028-001D09F2910A.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/003228A4-835B-DF11-AFBA-001D09F25217.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/002C392D-FF5A-DF11-A933-000423D99CEE.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/00237363-655B-DF11-BE4E-001D09F28EA3.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/0021FA08-775B-DF11-82A0-000423D6C8EE.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/00100ABE-0E5B-DF11-97AF-001D09F24664.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/00078CCF-155B-DF11-8A09-001D09F290CE.root',
'/store/data/Commissioning10/MinimumBias/RAW/v4/000/135/149/0005236B-715B-DF11-8EB5-0030486730C6.root'),            
)
process.DBService=cms.Service("DBService",
           authPath=cms.untracked.string('/afs/cern.ch/cms/DB/lumi')
)
process.lumiProducer=cms.EDProducer("LumiProducer",
   #connect=cms.string('frontier://cmsfrontier.cern.ch:8000/LumiPrep/CMS_LUMI_DEV_OFFLINE'),
   #connect=cms.string('oracle://cms_orcoff_prep/cms_lumi_dev_offline'),
   connect=cms.string('frontier://LumiPrep/CMS_LUMI_DEV_OFFLINE'),                                 
   lumiversion=cms.untracked.string('0001') 
)

process.test = cms.EDAnalyzer("TestLumiProducer")

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('testLumiProd-135149.root')
)
process.p1 = cms.Path(process.lumiProducer * process.test)
process.e = cms.EndPath(process.out)
