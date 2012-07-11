import FWCore.ParameterSet.Config as cms

process = cms.Process("dbsourcetest")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
#  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.DBService=cms.Service('DBService',
        authPath= cms.untracked.string('/afs/cern.ch/cms/lumi')       
)
process.source= cms.Source("PoolSource",
        processingMode=cms.untracked.string('RunsAndLumis'),        
        fileNames=cms.untracked.vstring(
        'file:/data/cmsdata/009F3522-D604-E111-A08D-003048F1183E.root')
)
process.LumiCorrectionSource=cms.ESSource("LumiCorrectionSource",
        connect=cms.string('frontier://LumiCalc/CMS_LUMI_PROD')
)
process.prod = cms.EDAnalyzer("TestLumiCorrectionSource")

process.p = cms.Path(process.prod)

