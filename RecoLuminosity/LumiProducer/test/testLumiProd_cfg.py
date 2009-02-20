
import FWCore.ParameterSet.Config as cms

process = cms.Process("REPACKER")

process.load("FWCore.MessageService.MessageLogger_cfi")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
#  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('LuminosityInfoRcd'),
        tag = cms.string('lumitest')
    )),
    connect = cms.string('sqlite_file:offlinelumi.db'),
    #connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_RUN_INFO'),                                  
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')                          
)
process.lumiProducer=cms.EDProducer("LumiProducer")
process.test = cms.EDAnalyzer("TestLumiProducer")

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string('testLumiProd.root')
)

process.p1 = cms.Path(process.lumiProducer * process.test)

process.e = cms.EndPath(process.out)
