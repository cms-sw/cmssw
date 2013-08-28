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
        record = cms.string('CSVM_T'),
        tag = cms.string('CSVM_T'),
        label = cms.string('CSVM_T')
    ),
cms.PSet(
        record = cms.string('CSVM_WP'),
        tag = cms.string('CSVM_WP'),
        label = cms.string('CSVM_WP')
    ))
                      
)


process.mywriter = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('combinedSecondaryVertex_Medium.txt'),
                                RecordPayload = cms.untracked.string('CSVM_T'),
                                RecordWP = cms.untracked.string('CSVM_WP')
                                )


process.p = cms.Path(process.mywriter)


