import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:DBs/NAME.db'


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    toPut = cms.VPSet(
#
# TEMPLATE
#
    cms.PSet(
        record = cms.string('NAME_T'),
        tag = cms.string('NAME_T'),
        label = cms.string('NAME_T')
    ),
cms.PSet(
        record = cms.string('NAME_WP'),
        tag = cms.string('NAME_WP'),
        label = cms.string('NAME_WP')


)
 )                     
)


#
# TEMPLATE
#

process.mywriterNAME = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('FILE'),
                                RecordPayload = cms.untracked.string('NAME_T'),
                                RecordWP = cms.untracked.string('NAME_WP')
                 )




process.p = cms.Path(
    process.mywriterNAME
   )
#


