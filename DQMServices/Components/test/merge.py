import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Merge")

options = VarParsing.VarParsing('analysis')
options.register('inFiles',
                 '',
                 VarParsing.VarParsing.multiplicity.list,
                 VarParsing.VarParsing.varType.string,
                 "input Files.")

options.parseArguments()





process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring(options.inFiles),
)
process.Merged = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string('Merged.root'),
)


process.outputPath = cms.EndPath(process.Merged)

process.DQMStore = cms.Service("DQMStore")

process.CPU = cms.Service("CPU")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")
process.Timing = cms.Service("Timing",
    summaryOnly = cms.untracked.bool(True)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


