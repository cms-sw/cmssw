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
        record = cms.string('TCHEM_T'),
        tag = cms.string('TCHEM_T'),
        label = cms.string('TCHEM_T')
    ),
cms.PSet(
        record = cms.string('TCHEM_WP'),
        tag = cms.string('TCHEM_WP'),
        label = cms.string('TCHEM_WP')
    ))
                      
)

process.mywriter = cms.EDFilter("PhysicsPerformanceDBWriterFromFile_WPandPayload",
                                inputTxtFile = cms.untracked.string('trackCountinhHighEfficiency_Medium.txt'),
                                RecordPayload = cms.untracked.string('TCHEM_T'),
                                RecordWP = cms.untracked.string('TCHEM_WP')
                                )



process.p = cms.Path(process.mywriter)


