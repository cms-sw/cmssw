import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('inputOTSensorList', 'CondTools/SiPhase2Tracker/data/bad_OT_sensors_example.txt',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string, 'text file with one OT sensor DetId per line')
options.register('inputITSensorList', 'CondTools/SiPhase2Tracker/data/bad_IT_sensors_example.txt',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string, 'text file with one IT sensor DetId per line')
options.register('outputName', 'BadSensorsByHand_T35_v0.db',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string, 'output sqlite file')
options.parseArguments()


process = cms.Process("WRITE")

process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T35', '')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPhase2BadSensorsByHandBuilder = dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1), reportEvery = cms.untracked.int32(1000)),
    SiPhase2BadSensorsByHandBuilder = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    DBParameters = cms.PSet(authenticationPath = cms.untracked.string('')),
    connect = cms.string('sqlite_file:' + options.outputName),
    toPut = cms.VPSet(
        cms.PSet(record = cms.string('SiPhase2OuterTrackerBadSensorRcd'), tag = cms.string('Phase2OTBadSensorsByHand_T35_v0')),
        cms.PSet(record = cms.string('SiPhase2InnerTrackerBadSensorRcd'), tag = cms.string('Phase2ITBadSensorsByHand_T35_v0')),
    )
)

process.otBadSensorsBuilder = cms.EDAnalyzer("SiPhase2BadSensorsByHandBuilder",
    Record = cms.string('SiPhase2OuterTrackerBadSensorRcd'),
    SinceAppendMode = cms.bool(True),
    IOVMode = cms.string('Run'),
    doStoreOnDB = cms.bool(True),
    badSensorListFile = cms.string(options.inputOTSensorList),
    printDebug = cms.untracked.bool(True),
)

process.itBadSensorsBuilder = cms.EDAnalyzer("SiPhase2BadSensorsByHandBuilder",
    Record = cms.string('SiPhase2InnerTrackerBadSensorRcd'),
    SinceAppendMode = cms.bool(True),
    IOVMode = cms.string('Run'),
    doStoreOnDB = cms.bool(True),
    badSensorListFile = cms.string(options.inputITSensorList),
    printDebug = cms.untracked.bool(True),
)

process.p = cms.Path(process.otBadSensorsBuilder + process.itBadSensorsBuilder)
