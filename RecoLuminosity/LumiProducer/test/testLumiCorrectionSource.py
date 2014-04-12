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

process.source= cms.Source("PoolSource",
        processingMode=cms.untracked.string('RunsAndLumis'),        
        fileNames=cms.untracked.vstring(
        #'file:/data/cmsdata/200786/FC7E661B-C3E8-E111-A23E-003048D2BDD8.root'
        'file:testLumiProd-200786.root'
        )
)

process.LumiCorrectionSource=cms.ESSource("LumiCorrectionSource",
        authpath=cms.untracked.string('/afs/cern.ch/cms/lumi/DB'),
        connect=cms.string('oracle://cms_orcon_adg/cms_lumi_prod')
        #connect=cms.string('frontier://LumiCalc/CMS_LUMI_PROD'),
        #normtag=cms.untracked.string('HFV2a')
        #datatag=cms.untracked.string('v3')
)
process.prod = cms.EDAnalyzer("TestLumiCorrectionSource")

process.p = cms.Path(process.prod)

