import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:../PhysicsPerformance.db'

#process.load ("RecoBTag.PerformanceDB.BtagPerformanceESProducer_cfi")
#
# change inside the source
#



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    )


process.source = cms.Source(
    "PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(   


"file:PM_pattuple_8.root"



    )
)




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

process.demo = cms.EDAnalyzer('PATJetAnalyzer',
                              CalibrationForBEfficiency = cms.string('Ptrel_JPL'),
                              CalibrationForMistag = cms.string('Mistag_JPL'),
                              PATJetCollection = cms.string('selectedLayer1Jets')
                              )

process.p = cms.Path(process.demo)

#

