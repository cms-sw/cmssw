import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load("CondCore.DBCommon.CondDBCommon_cfi") 
process.load ("RecoBTag.PerformanceDB.PoolBTagPerformanceDB100426") 
process.load ("RecoBTag.PerformanceDB.BTagPerformanceDB100426") 
process.PoolDBESSource.connect = 'frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'
process.PoolDBESSource.toGet = cms.VPSet(
    cms.PSet(
        record = cms.string('PerformanceWPRecord'),
        tag = cms.string('BTagPTRELSSVMwp_v1_offline'),
        label = cms.untracked.string('BTagPTRELSSVMwp_v1_offline')
        ),
    cms.PSet(
        record = cms.string('PerformancePayloadRecord'),
        tag = cms.string('BTagPTRELSSVMtable_v1_offline'),
        label = cms.untracked.string('BTagPTRELSSVMtable_v1_offline')
        ),
    cms.PSet(
        record = cms.string('PerformanceWPRecord'),
        tag = cms.string('BTagMISTAGSSVMwp_v1_offline'),
        label = cms.untracked.string('BTagMISTAGSSVMwp_v1_offline')
        ),
    cms.PSet(
        record = cms.string('PerformancePayloadRecord'),
        tag = cms.string('BTagMISTAGSSVMtable_v1_offline'),
        label = cms.untracked.string('BTagMISTAGSSVMtable_v1_offline')
        )
    )


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.TFileService = cms.Service("TFileService", fileName = cms.string("performance_ssvm.root") )


process.myrootwriter = cms.EDAnalyzer("BTagPerformaceRootProducerFromSQLITE",
                                  names = cms.vstring('PTRELSSVM', 'MISTAGSSVM'),
                                  index = cms.uint32(1001)
                                  )


process.p = cms.Path(process.myrootwriter)
