import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing('analysis')

options.register('rootFile',
                 "", # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Input file.")

options.register('runInputDir',
                 '/tmp/test/', # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Directory where the DQM files will appear.")

options.parseArguments()

process = cms.Process('daqstreamer')

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1), # every!
    limit = cms.untracked.int32(-1)     # no limit!
    )
process.MessageLogger.cerr_stats.threshold = 'INFO' # also info in statistics

# read back the trigger decisions
process.source = cms.Source('PoolSource',
    #fileNames = cms.untracked.vstring('file:/build1/micius/run198487_SinglePhoton.root'),
    fileNames = cms.untracked.vstring(options.rootFile),
    noEventSort = cms.untracked.bool(False),
)

process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(-1)
)

# define the PoolOutputModule
process.poolOutput = cms.OutputModule('DQMStreamerOutputModule',
    runInputDir = cms.untracked.string(options.runInputDir),
    streamLabel = cms.untracked.string("_streamA"),
)

process.output = cms.EndPath(process.poolOutput)

