import FWCore.ParameterSet.Config as cms

process = cms.Process("tester")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(285243))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

#process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_cfi")

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string('sqlite:l1config.db')
#CondDB.connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
#
process.l1conddb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet   = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TCaloStage2ParamsRcd'),
                 tag = cms.string("L1TCaloParams_Stage2v0_hlt")
            )
       )
)

process.l1cpu = cms.EDAnalyzer("L1TCaloParamsUpdater")

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
outputDB = cms.Service("PoolDBOutputService",
                       CondDBSetup,
                       connect = cms.string('sqlite_file:l1configTweak.db'),
                       toPut   = cms.VPSet(
                           cms.PSet(
                               record = cms.string('L1TCaloStage2ParamsTweakedRcd'),
                               tag = cms.string("L1TCaloStage2Params_Stage2v0_tweak_hlt")
                           )
                       )
)
outputDB.DBParameters.authenticationPath = '.'
process.add_(outputDB)

process.p = cms.Path(process.l1cpu)

