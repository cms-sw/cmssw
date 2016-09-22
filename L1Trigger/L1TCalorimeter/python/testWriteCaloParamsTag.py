import FWCore.ParameterSet.Config as cms

process = cms.Process('L1TEST')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('L1-O2O')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.source = cms.Source('EmptySource')

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag.connect = cms.string('frontier://FrontierProd/CMS_COND_31X_GLOBALTAG')
process.GlobalTag.globaltag = cms.string('POSTLS162_V2::All')

# New parameters
process.load('L1Trigger.L1TCalorimeter.caloStage1Params_cfi')
process.caloParamsWriter = cms.EDAnalyzer("CaloParamsWriter")

process.p = cms.Path(
    process.caloParamsWriter
)

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondDBSetup,
    connect = cms.string('sqlite_file:l1config.db'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('L1TCaloParamsRcd'),
            tag = cms.string('L1TCaloParamsRcd_Testing')
        )
    )
)
