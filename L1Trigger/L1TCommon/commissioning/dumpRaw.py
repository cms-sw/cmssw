import FWCore.ParameterSet.Config as cms

# options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')

options.register('stream',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Use stream file")

options.register('fed',
                 813,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "FED ID")

options.parseArguments()

print options.inputFiles

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

if (options.stream):
    process.source = cms.Source("NewEventStreamFileReader",
                                fileNames = cms.untracked.vstring(options.inputFiles)
                                )
else:
    process.source = cms.Source("PoolSource",
                                fileNames=cms.untracked.vstring(options.inputFiles)
                                )

process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("rawDataCollector"),
    feds = cms.untracked.vint32 ( options.fed ),
    dumpPayload = cms.untracked.bool ( True )
)

process.path = cms.Path(
    process.dumpRaw
)

#process.out = cms.OutputModule(
#            "PoolOutputModule",
#            fileName = cms.untracked.string("output.root")
#)

#process.outpath = cms.EndPath(
#    process.out
#)
