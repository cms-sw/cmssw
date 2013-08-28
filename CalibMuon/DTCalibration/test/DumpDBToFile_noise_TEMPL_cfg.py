import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpDBToFile")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.get = cms.EDAnalyzer("DTTtrigPrint")

process.calibDB = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    authenticationMethod = cms.untracked.uint32(0),
    toGet = cms.VPSet(cms.PSet(
    	# Noise
        record = cms.string('DTStatusFlagRcd'),
        tag = cms.string('noise')
    )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPL/noise/noise_RUNNUMBERTEMPLATE.db')
)

process.dumpToFile = cms.EDAnalyzer("DumpDBToFile",
    #Choose what database you want to write
    dbToDump = cms.untracked.string('NoiseDB'),
    dbLabel = cms.untracked.string(''),
    calibFileConfig = cms.untracked.PSet(
    # Noise    
    nFields = cms.untracked.int32(7),
    calibConstGranularity = cms.untracked.string('byWire')
    ),
    outputFileName = cms.untracked.string('/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPL/noise/noise_RUNNUMBERTEMPLATE.txt')
)

process.p = cms.Path(process.dumpToFile)


