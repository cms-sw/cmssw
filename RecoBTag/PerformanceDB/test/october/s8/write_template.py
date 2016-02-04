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
# TEMPLATE
#
    cms.PSet(
        record = cms.string('NAMETEMPLATE_T'),
        tag = cms.string('NAMETEMPLATE_T'),
        label = cms.string('NAMETEMPLATE_T')
    ),
cms.PSet(
        record = cms.string('NAMETEMPLATE_WP'),
        tag = cms.string('NAMETEMPLATE_WP'),
        label = cms.string('NAMETEMPLATE_WP')


)
 )                     
)


#
# TEMPLATE
#

process.mywriterTEMPLATE = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('resultTEMPLATE.txt'),
                                RecordPayload = cms.untracked.string('NAMETEMPLATE_T'),
                                RecordWP = cms.untracked.string('NAMETEMPLATE_WP')
                 )




process.p = cms.Path(
    process.mywriterTEMPLATE
   )
#


