import FWCore.ParameterSet.Config as cms

process = cms.Process("read")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#    process.CondDBSetup,
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('L1RPCHwConfigRcd'),
#        tag = cms.string('L1RPCHwConfig_STARTUP')
#    )),
#    connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_30X_RPC')
#)
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_31X::All'

process.tmf = cms.EDFilter("TestHWConfig")

process.p = cms.Path(process.tmf)


