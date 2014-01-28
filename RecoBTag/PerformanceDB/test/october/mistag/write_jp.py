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
        record = cms.string('MISTAGJPL_T'),
        tag = cms.string('MISTAGJPL_T'),
        label = cms.string('MISTAGJPL_T')
    ),
cms.PSet(
        record = cms.string('MISTAGJPL_WP'),
        tag = cms.string('MISTAGJPL_WP'),
        label = cms.string('MISTAGJPL_WP')
    ),
cms.PSet(
        record = cms.string('MISTAGJPM_T'),
        tag = cms.string('MISTAGJPM_T'),
        label = cms.string('MISTAGJPM_T')
    ),
cms.PSet(
        record = cms.string('MISTAGJPM_WP'),
        tag = cms.string('MISTAGJPM_WP'),
        label = cms.string('MISTAGJPM_WP')
    ),
cms.PSet(
        record = cms.string('MISTAGJPT_T'),
        tag = cms.string('MISTAGJPT_T'),
        label = cms.string('MISTAGJPT_T')
    ),
cms.PSet(
        record = cms.string('MISTAGJPT_WP'),
        tag = cms.string('MISTAGJPT_WP'),
        label = cms.string('MISTAGJPT_WP')
    ),

)
                      
)

process.mywriterl = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('table_mistag_jpl.txt'),
                                RecordPayload = cms.untracked.string('MISTAGJPL_T'),
                                RecordWP = cms.untracked.string('MISTAGJPL_WP')
                 )

process.mywriterm = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('table_mistag_jpm.txt'),
                                RecordPayload = cms.untracked.string('MISTAGJPM_T'),
                                RecordWP = cms.untracked.string('MISTAGJPM_WP')
                 )

process.mywritert = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('table_mistag_jpt.txt'),
                                RecordPayload = cms.untracked.string('MISTAGJPT_T'),
                                RecordWP = cms.untracked.string('MISTAGJPT_WP')
                 )



process.p = cms.Path(process.mywriterl+ process.mywriterm+process.mywritert)
#


