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
        record = cms.string('JPT_T'),
        tag = cms.string('JPT_T'),
        label = cms.string('JPT_T')
    ),
cms.PSet(
        record = cms.string('JPT_WP'),
        tag = cms.string('JPT_WP'),
        label = cms.string('JPT_WP')
    ))
                      
)

process.mywriter = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('jetProbability_Tight.txt'),
                                RecordPayload = cms.untracked.string('JPT_T'),
                                RecordWP = cms.untracked.string('JPT_WP')
                                )



process.p = cms.Path(process.mywriter)


