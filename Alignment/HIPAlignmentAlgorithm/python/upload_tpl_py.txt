import FWCore.ParameterSet.Config as cms
process = cms.Process("Alignment")

# "including" common configuration
<COMMON>

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:<PATH>/alignments.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TrackerAlignmentRcd'),
        tag = cms.string('Alignments')
    ), 
        cms.PSet(
            record = cms.string('TrackerAlignmentErrorRcd'),
            tag = cms.string('AlignmentErrors')
        ))
)

process.AlignmentProducer.algoConfig.outpath = '<PATH>/'
process.AlignmentProducer.saveToDB = True


