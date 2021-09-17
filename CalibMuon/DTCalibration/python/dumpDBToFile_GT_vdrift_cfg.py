import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpDBToFile")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ''

process.load("CondCore.CondDB.CondDB_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.dumpToFile = cms.EDAnalyzer("DumpDBToFile",
    # Choose what database you want to write
    dbToDump = cms.untracked.string('VDriftDB'),
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(5),
        # VDrift & TTrig
        calibConstGranularity = cms.untracked.string('bySL')
    ),
    outputFileName = cms.untracked.string('vDrift.txt')
)

process.p = cms.Path(process.dumpToFile)
