# cfg file to test L1 GT trigger menu read from DB

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process("L1GtTriggerMenuTest")
process.l1GtTriggerMenuTest = cms.EDAnalyzer("L1GtTriggerMenuTester")

# number of events and source
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

# configuration
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

# L1 db
from CondCore.DBCommon.CondDBCommon_cfi import *
from CondTools.L1Trigger.L1SubsystemParams_cfi import initL1Subsystems
initL1Subsystems()

process.l1pooldb = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    toGet = initL1Subsystems.params.recordInfo
)
process.l1pooldb.connect = cms.string('sqlite_file:l1config.db')

# Other statements
process.GlobalTag.globaltag = 'IDEAL_V5::All'

# path to be run
process.p = cms.Path(process.l1GtTriggerMenuTest)

# services

# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules = ['l1GtTriggerMenuTest']
process.MessageLogger.cout = cms.untracked.PSet(
    INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
    ),
    threshold = cms.untracked.string('DEBUG'), ## DEBUG 

    DEBUG = cms.untracked.PSet( ## DEBUG, all messages  

        limit = cms.untracked.int32(-1)
    )
)


