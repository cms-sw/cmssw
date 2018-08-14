import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpDBToFile")

process.load("CondCore.CondDB.CondDB_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.calibDB = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
    	# Noise
        record = cms.string('DTStatusFlagRcd'),
        tag = cms.string('noise')
    )),
)
process.calibDB.connect = cms.string('sqlite_file:noise.db')

process.dumpToFile = cms.EDAnalyzer("DumpDBToFile",
    # Choose what database you want to write
    dbToDump = cms.untracked.string('NoiseDB'),
    dbLabel = cms.untracked.string(''),
    calibFileConfig = cms.untracked.PSet(
        # Noise
        nFields = cms.untracked.int32(7),
        calibConstGranularity = cms.untracked.string('byWire')
    ),
    outputFileName = cms.untracked.string('noise.txt')
)

process.p = cms.Path(process.dumpToFile)
