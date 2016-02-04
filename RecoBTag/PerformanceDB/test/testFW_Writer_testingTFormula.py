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
        record = cms.string('pippo1'),
        tag = cms.string('TEST_TF_1'),
        label = cms.string('TEST_TF_1')
    ),
cms.PSet(
        record = cms.string('pippo2'),
        tag = cms.string('TEST_TF_1_WP'),
        label = cms.string('TEST_TF_1_WP')
    ))
                      
)

process.mywriter = cms.EDFilter("PhysicsPerformanceDBWriterTFormula_fromfile_WPandPL",
                                inputTxtFile = cms.untracked.string('tformulainput.txt'),
                                RecordPayload = cms.untracked.string('pippo1'),
                                RecordWP = cms.untracked.string('pippo2')
                                )


process.p = cms.Path(process.mywriter)


