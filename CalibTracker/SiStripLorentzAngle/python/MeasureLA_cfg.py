import FWCore.ParameterSet.Config as cms

process = cms.Process("MACRO")
process.add_(cms.Service("MessageLogger"))
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

#make sqlite file, remember name.

process.measurement = cms.ESProducer(
    "sistrip::MeasureLA",
    inputFiles = cms.vstring([]),
    byLayer = cms.bool(True),
    byModule = cms.bool(True),
    outputHistograms = cms.string(""),
    outputSqliteFile = cms.string(""),
    defaultSqliteFile = cms.string("")
    )
