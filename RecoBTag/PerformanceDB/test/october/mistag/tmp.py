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
    toPut = cms.VPSet(
#
# TCHPT
#
    cms.PSet(
        record = cms.string('MISTAGTCHPT_T'),
        tag = cms.string('MISTAGTCHPT_T'),
        label = cms.string('MISTAGTCHPT_T')
    ),
cms.PSet(
        record = cms.string('MISTAGTCHPT_WP'),
        tag = cms.string('MISTAGTCHPT_WP'),
        label = cms.string('MISTAGTCHPT_WP')


)
 )                     
)


#
# TCHPT
#

process.mywriterTCHPT = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('resultTCHPT.txt'),
                                RecordPayload = cms.untracked.string('MISTAGTCHPT_T'),
                                RecordWP = cms.untracked.string('MISTAGTCHPT_WP')
                 )




process.p = cms.Path(
    process.mywriterTCHPT
   )
#


