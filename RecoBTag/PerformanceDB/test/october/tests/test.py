import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:../PhysicsPerformance.db'

#process.load ("RecoBTag.PerformanceDB.BtagPerformanceESProducer_cfi")
#
# change inside the source
#

BtagPerformanceESProducer_mistagJPL = cms.ESProducer("BtagPerformanceESProducer",
                                                     # this is what it makes available
                                                     ComponentName = cms.string('Mistag_JPL'),
                                                     # this is where it gets the payload from                                                
                                                     PayloadName = cms.string('MISTAGJPL_T'),
                                                     WorkingPointName = cms.string('MISTAGJPL_WP')
                                                     )
BtagPerformanceESProducer_mistagJPM = cms.ESProducer("BtagPerformanceESProducer",
                                                     # this is what it makes available
                                                     ComponentName = cms.string('Mistag_JPM'),
                                                     # this is where it gets the payload from                                                
                                                     PayloadName = cms.string('MISTAGJPM_T'),
                                                     WorkingPointName = cms.string('MISTAGJPM_WP')
                                                     )
BtagPerformanceESProducer_mistagJPT = cms.ESProducer("BtagPerformanceESProducer",
                                                     # this is what it makes available
                                                     ComponentName = cms.string('Mistag_JPT'),
                                                     # this is where it gets the payload from                                                
                                                     PayloadName = cms.string('MISTAGJPT_T'),
                                                     WorkingPointName = cms.string('MISTAGJPT_WP')
                                                     )




process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      process.CondDBCommon,
                                      toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('MISTAGJPT_T'),
    label = cms.untracked.string('MISTAGJPT_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('MISTAGJPL_T'),
    label = cms.untracked.string('MISTAGJPL_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('MISTAGJPM_T'),
    label = cms.untracked.string('MISTAGJPM_T')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('MISTAGJPT_WP'),
    label = cms.untracked.string('MISTAGJPT_WP')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('MISTAGJPL_WP'),
    label = cms.untracked.string('MISTAGJPL_WP')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('MISTAGJPM_WP'),
    label = cms.untracked.string('MISTAGJPM_WP')
    )
    )
)

process.demo = cms.EDAnalyzer('TestOctoberExe',
                              CalibrationForBEfficiency = cms.string('Ptrel_JPL'),
                              CalibrationForMistag = cms.string('Mistag_JPL')
                              )

process.p = cms.Path(process.demo)

#

