import FWCore.ParameterSet.Config as cms

process = cms.Process("read")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.GlobalPositionSource = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    # Reading from oracle (instead of Frontier) needs the following shell variable setting (in zsh):
    # export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
    # string connect = "oracle://cms_orcoff_int2r/CMS_COND_ALIGNMENT"
    # untracked uint32 authenticationMethod = 1
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('GlobalPositionRcd'),
        tag = cms.string('IdealGeometry')
    )),
    connect = cms.string('sqlite_file:output.db')
)

process.GlobalPositionRcdRead = cms.EDAnalyzer("GlobalPositionRcdRead")

process.p = cms.Path(process.GlobalPositionRcdRead)


