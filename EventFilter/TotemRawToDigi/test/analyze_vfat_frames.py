import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("AnalyzeVFATFrames")

# default options
options = VarParsing.VarParsing ('analysis')
options.inputFiles= 'file:/afs/cern.ch/user/j/jkaspar/public/run268608_ls0001_streamA_StorageManager.root',
options.outputFile = 'this_is_not_used'
options.maxEvents = 10

# parse command-line options
options.parseArguments()

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

# raw data source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# frame analyzer
process.totemVFATFrameAnalyzer = cms.EDAnalyzer("TotemVFATFrameAnalyzer",
    rawDataTag = cms.InputTag("rawDataCollector"),
    fedIds = cms.vuint32(578, 579, 580),
    RawUnpacking = cms.PSet()
)

# execution configuration
process.p = cms.Path(
    process.totemVFATFrameAnalyzer
)
