import os
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')
options.register('inputModuleList', os.path.join(os.environ['CMSSW_BASE'], 'src/CondTools/SiPhase2Tracker/data/bad_modules_example.txt'),
                 VarParsing.multiplicity.singleton, VarParsing.varType.string, 'text file with one DetId per line, each expanded to its full module')
options.register('inputSensorList', os.path.join(os.environ['CMSSW_BASE'], 'src/CondTools/SiPhase2Tracker/data/bad_sensors_example.txt'),
                 VarParsing.multiplicity.singleton, VarParsing.varType.string, 'text file with one DetId per line, each written as it is')
options.register('outputName', 'BadComponentsByHand_v0.db',
                 VarParsing.multiplicity.singleton, VarParsing.varType.string, 'output sqlite file')
options.parseArguments()


process = cms.Process("WRITE")

process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T35', '')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPhase2BadComponentsByHandBuilder = dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1), reportEvery = cms.untracked.int32(1000)),
    SiPhase2BadComponentsByHandBuilder = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
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
        cms.PSet(record = cms.string('SiPhase2OuterTrackerBadModuleRcd'), tag = cms.string('Phase2OTBadComponentsByHand_T35_v0')),
        cms.PSet(record = cms.string('SiPhase2InnerTrackerBadModuleRcd'), tag = cms.string('Phase2ITBadComponentsByHand_T35_v0')),
    )
)

process.otBadComponentsBuilder = cms.EDAnalyzer("SiPhase2BadComponentsByHandBuilder",
    Record = cms.string('SiPhase2OuterTrackerBadModuleRcd'),
    SinceAppendMode = cms.bool(True),
    IOVMode = cms.string('Run'),
    doStoreOnDB = cms.bool(True),
    badModuleListFile = cms.untracked.string(options.inputModuleList),
    badSensorListFile = cms.untracked.string(options.inputSensorList),
    targetRecord = cms.untracked.string("SiPhase2OuterTrackerBadModuleRcd"),
    printDebug = cms.untracked.bool(True),
)

process.itBadComponentsBuilder = cms.EDAnalyzer("SiPhase2BadComponentsByHandBuilder",
    Record = cms.string('SiPhase2InnerTrackerBadModuleRcd'),
    SinceAppendMode = cms.bool(True),
    IOVMode = cms.string('Run'),
    doStoreOnDB = cms.bool(True),
    badModuleListFile = cms.untracked.string(options.inputModuleList),
    badSensorListFile = cms.untracked.string(options.inputSensorList),
    targetRecord = cms.untracked.string("SiPhase2InnerTrackerBadModuleRcd"),
    printDebug = cms.untracked.bool(True),
)

process.p = cms.Path(process.otBadComponentsBuilder + process.itBadComponentsBuilder)
