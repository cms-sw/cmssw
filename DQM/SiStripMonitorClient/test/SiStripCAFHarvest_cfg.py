import FWCore.ParameterSet.Config as cms

process = cms.Process("CAFHarvestingJob")

#-------------------------------------------------
## Empty Event Source
#-------------------------------------------------
process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(58733),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(58733),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#-------------------------------------------------
## Message Logger
#-------------------------------------------------
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('error.log', 
        'cout')
)

#-------------------------------------------------
# DQM Services
#-------------------------------------------------

process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(1)
)

process.qTester = cms.EDFilter("QualityTester",
    qtestOnEndJob = cms.untracked.bool(True),
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorClient/data/sistrip_qualitytest_config.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

#-------------------------------------------------
## DQM Harvesting
#-------------------------------------------------
process.dqmHarvesing = cms.EDFilter("SiStripOfflineDQM",
    CreateSummary = cms.untracked.bool(True),
    InputFileName = cms.untracked.string('DQM_SiStrip_R000058733-standAlone.root'),
    OutputFileName = cms.untracked.string('DQM_SiStrip_R000058289_CAF.root'),
    GlobalStatusFilling = cms.untracked.int32(1)
)

#-------------------------------------------------
## Scheduling
#-------------------------------------------------
process.p = cms.Path(process.qTester*process.dqmHarvesing)

