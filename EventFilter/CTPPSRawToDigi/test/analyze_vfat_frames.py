import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("AnalyzeVFATFrames")

# default options
options = VarParsing.VarParsing ('analysis')
options.inputFiles= 'file:/eos/cms/store/group/dpg_ctpps/comm_ctpps/TotemTiming/Minidaq/306/run312306_ls0015_streamA_StorageManager.dat',
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
process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(options.inputFiles)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# frame analyzer
process.totemVFATFrameAnalyzer = cms.EDAnalyzer("TotemVFATFrameAnalyzer",
    rawDataTag = cms.InputTag("rawDataCollector"),
    fedIds = cms.vuint32(587),
    RawUnpacking = cms.PSet()
)

# execution configuration
process.p = cms.Path(
    process.totemVFATFrameAnalyzer
)

