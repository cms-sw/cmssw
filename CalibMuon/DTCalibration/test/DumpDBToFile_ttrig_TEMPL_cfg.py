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
        # VDrift
        #string record = "DTMtimeRcd"
        #string tag ="vDrift"
        # TZero
        #string record = "DTT0Rcd" 
        #string tag = "t0"
        #string tag = "t0_GRUMM"
        # TTrig
        record = cms.string('DTTtrigRcd'),
        tag = cms.string('ttrig')
    )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPL/ttrig/ttrig_DUMPDBTEMPL_RUNNUMBERTEMPLATE.db')
)

process.dumpToFile = cms.EDAnalyzer("DumpDBToFile",
    #Choose what database you want to write
    #untracked string dbToDump = "VDriftDB"
    #untracked string dbToDump = "TZeroDB"
    dbToDump = cms.untracked.string('TTrigDB'),
    dbLabel = cms.untracked.string(''),
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(5),
        # VDrift & TTrig
        calibConstGranularity = cms.untracked.string('bySL')
    ),
    outputFileName = cms.untracked.string('/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONCALIB/DTCALIB/RUNPERIODTEMPL/ttrig/ttrig_DUMPDBTEMPL_RUNNUMBERTEMPLATE.txt')
)

process.p = cms.Path(process.dumpToFile)
