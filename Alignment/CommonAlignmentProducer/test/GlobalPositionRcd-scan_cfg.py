
import FWCore.ParameterSet.Config as cms

process = cms.Process("read")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000000)


process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(1), # do not change!
                            firstRun = cms.untracked.uint32(1)
                            )
process.maxEvents = cms.untracked.PSet(
    # with firstRun = 1 and numberEventsInRun = 1 above, this number meanse that we...
    input = cms.untracked.int32(200000)  # ...look for runs from 1 to this number
    )

process.GlobalPositionSource = cms.ESSource(
    "PoolDBESSource",
    process.CondDBSetup,
    # Reading from oracle (instead of Frontier) needs the following shell variable setting (in zsh):
    # export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
    # string connect = "oracle://cms_orcoff_int2r/CMS_COND_ALIGNMENT"
    # untracked uint32 authenticationMethod = 1
    toGet = cms.VPSet(cms.PSet(record = cms.string('GlobalPositionRcd'),
##                              tag = cms.string('GlobalAlignment_2009_v1_express')
#                               tag = cms.string('GlobalPosition'))
                               tag = cms.string('GlobalAlignment_v2_offline'))
                      ),
    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_ALIGNMENT')
#    connect = cms.string('sqlite_file:output.db')
    )
process.GlobalPositionRcdScan = cms.EDAnalyzer(
    "GlobalPositionRcdScan",
    # defining how the rotation matrix is printed: 'euler', 'align', 'matrix' or 'all'
    rotation = cms.string('all')
    )

process.p = cms.Path(process.GlobalPositionRcdScan)


