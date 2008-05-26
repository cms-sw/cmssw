import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string("")
process.CondDBCommon.DBParameters.messageLevel = 0

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
   connect = cms.string(''),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    toGet = cms.VPSet(
        cms.PSet(
        connect = cms.untracked.string('oracle://cms_orcoff_prod/CMS_COND_20X_DT'),
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('tTrig_GRUMM_080313_hlt'),
        label = cms.untracked.string('t2')
        ), 
        cms.PSet(
        connect = cms.untracked.string('sqlite_file:orconGRUMM_200p9.db'),
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('tTrig_GRUMM_080313'),
        label = cms.untracked.string('t1')
        )
     )
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(3),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        data = cms.vstring('DTTtrig/t1', 'DTTtrig/t2')
    )),
    verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.get)



