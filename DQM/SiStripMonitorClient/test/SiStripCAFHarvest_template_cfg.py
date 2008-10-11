import FWCore.ParameterSet.Config as cms

process = cms.Process("CAFHarvestingJob")

#-------------------------------------------------
## Empty Event Source
#-------------------------------------------------
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(xRUN_NUMBERx),
    lastRun  = cms.untracked.uint32(xRUN_NUMBERx),
    interval = cms.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#-------------------------------------------------
## Message Logger
#-------------------------------------------------
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
)

#-------------------------------------------------
# DQM Services
#-------------------------------------------------

process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose           = cms.untracked.int32(0)
)

process.qTester = cms.EDFilter("QualityTester",
    qtestOnEndJob           = cms.untracked.bool(True),
    qtList                  = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
    QualityTestPrescaler    = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

#-------------------------------------------------
## DQM Harvesting
#-------------------------------------------------
process.dqmHarvesing = cms.EDFilter("SiStripOfflineDQM",
    CreateSummary       = cms.untracked.bool(True),
    InputFileName       = cms.untracked.string('xMERGED_INPUT_FILEx'),
    OutputFileName      = cms.untracked.string('xMERGED_OUTPUT_FILEx'),
    GlobalStatusFilling      = cms.untracked.int32(1)
)

#-------------------------------------------------
## Scheduling
#-------------------------------------------------
process.p = cms.Path(
    process.qTester      *
    process.dqmHarvesing
)

