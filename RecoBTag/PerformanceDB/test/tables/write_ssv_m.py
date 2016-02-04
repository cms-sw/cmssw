import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:PhysicsPerformance.db'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SSVM_T'),
        tag = cms.string('SSVM_T'),
        label = cms.string('SSVM_T')
    ),
cms.PSet(
        record = cms.string('SSVM_WP'),
        tag = cms.string('SSVM_WP'),
        label = cms.string('SSVM_WP')
    ))
                      
)

process.mywriter = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('simpleSecondaryVertex_Medium.txt'),
                                RecordPayload = cms.untracked.string('SSVM_T'),
                                RecordWP = cms.untracked.string('SSVM_WP')
                                )



process.p = cms.Path(process.mywriter)


