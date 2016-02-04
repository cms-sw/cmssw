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
        record = cms.string('SSVT_T'),
        tag = cms.string('SSVT_T'),
        label = cms.string('SSVT_T')
    ),
cms.PSet(
        record = cms.string('SSVT_WP'),
        tag = cms.string('SSVT_WP'),
        label = cms.string('SSVT_WP')
    ))
                      
)

process.mywriter = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('simpleSecondaryVertex_Tight.txt'),
                                RecordPayload = cms.untracked.string('SSVT_T'),
                                RecordWP = cms.untracked.string('SSVT_WP')
                                )



process.p = cms.Path(process.mywriter)


