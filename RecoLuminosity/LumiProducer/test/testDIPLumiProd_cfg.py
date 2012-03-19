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
#process.source = cms.Source("EmptySource",
#        numberEventsInRun = cms.untracked.uint32(10),
#        firstRun = cms.untracked.uint32(1),
#        numberEventsInLuminosityBlock = cms.untracked.uint32(1),
#        firstLuminosityBlock = cms.untracked.uint32(1)
#)
process.DBService=cms.Service('DBService',
        authPath= cms.untracked.string('/data/cmsdata')       
)
process.source= cms.Source("PoolSource",
        processingMode=cms.untracked.string('RunsAndLumis'),        
        fileNames=cms.untracked.vstring(
        'file:/data/cmsdata/009F3522-D604-E111-A08D-003048F1183E.root')
)
process.DIPLumiProducer=cms.ESSource("DIPLumiProducer",
        connect=cms.string('oracle://cms_orcoff_prod/cms_runtime_logger')
)
process.prod = cms.EDAnalyzer("TestDIPLumiProducer")

process.p = cms.Path(process.prod)

