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
# SSVT
#
    cms.PSet(
        record = cms.string('PTRELSSVT_T'),
        tag = cms.string('PTRELSSVT_T'),
        label = cms.string('PTRELSSVT_T')
    ),
cms.PSet(
        record = cms.string('PTRELSSVT_WP'),
        tag = cms.string('PTRELSSVT_WP'),
        label = cms.string('PTRELSSVT_WP')


)
 )                     
)


#
# SSVT
#

process.mywriterSSVT = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('resultSSVT.txt'),
                                RecordPayload = cms.untracked.string('PTRELSSVT_T'),
                                RecordWP = cms.untracked.string('PTRELSSVT_WP')
                 )




process.p = cms.Path(
    process.mywriterSSVT
   )
#


