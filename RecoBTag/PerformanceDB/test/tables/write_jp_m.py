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
        record = cms.string('JPM_T'),
        tag = cms.string('JPM_T'),
        label = cms.string('JPM_T')
    ),
cms.PSet(
        record = cms.string('JPM_WP'),
        tag = cms.string('JPM_WP'),
        label = cms.string('JPM_WP')
    ))
                      
)

process.mywriter = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('jetProbability_Medium.txt'),
                                RecordPayload = cms.untracked.string('JPM_T'),
                                RecordWP = cms.untracked.string('JPM_WP')
                                )



process.p = cms.Path(process.mywriter)


