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
        record = cms.string('JPL_T'),
        tag = cms.string('JPL_T'),
        label = cms.string('JPL_T')
    ),
cms.PSet(
        record = cms.string('JPL_WP'),
        tag = cms.string('JPL_WP'),
        label = cms.string('JPL_WP')
    ))
                      
)

process.mywriter = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('jetProbability_Loose.txt'),
                                RecordPayload = cms.untracked.string('JPL_T'),
                                RecordWP = cms.untracked.string('JPL_WP')
                 )



process.p = cms.Path(process.mywriter)


