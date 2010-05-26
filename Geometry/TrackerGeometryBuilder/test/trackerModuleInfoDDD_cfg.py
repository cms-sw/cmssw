import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
# empty input service, fire 10 events
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Choose Tracker Geometry
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = 'MC_31X_V8::All'

process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")
process.preferFakeAlign = cms.ESPrefer("FakeAlignmentSource") 

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.out = cms.OutputModule("AsciiOutputModule")

process.prod = cms.EDAnalyzer("ModuleInfo",
    fromDDD = cms.bool(True),
    printDDD = cms.untracked.bool(False)
)

process.p1 = cms.Path(process.prod)
process.ep = cms.EndPath(process.out)


