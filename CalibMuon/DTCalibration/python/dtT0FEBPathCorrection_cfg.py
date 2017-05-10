import FWCore.ParameterSet.Config as cms

process = cms.Process("DTT0Correction")

process.load("CalibMuon.DTCalibration.messageLoggerDebug_cff")
process.MessageLogger.debugModules = cms.untracked.vstring('dtT0FEBPathCorrection')

#process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = ''
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.source = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    authenticationMethod = cms.untracked.uint32(0),
    toGet = cms.VPSet(cms.PSet(
        # TZero
        record = cms.string("DTT0Rcd"),
        tag = cms.string("t0")
    )),
    connect = cms.string('sqlite_file:t0_input.db')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:t0_febcorrected.db'),
    authenticationMethod = cms.untracked.uint32(0),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DTT0Rcd'),
        tag = cms.string('t0')
    ))
)

#process.load("CalibMuon.DTCalibration.dtT0FEBPathCorrection_cfi")
process.dtT0FEBPathCorrection = cms.EDAnalyzer("DTT0Correction",
    correctionAlgo = cms.string('DTT0FEBPathCorrection'),
    correctionAlgoConfig = cms.PSet(
        # Format "wheel station sector" (e.g. "-1 3 10")
        calibChamber = cms.string('All'),
    )
)

process.dtT0FEBPathCorrection.correctionAlgoConfig.calibChamber = 'All'

process.p = cms.Path(process.dtT0FEBPathCorrection)
