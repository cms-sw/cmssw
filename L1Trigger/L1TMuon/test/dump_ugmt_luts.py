import FWCore.ParameterSet.Config as cms

process = cms.Process("L1MicroGMTEmulator")

process.load("FWCore.MessageService.MessageLogger_cfi")


process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

process.load('L1Trigger.L1TMuon.fakeGmtParams_cff')

process.dumper = cms.EDAnalyzer("L1TMicroGMTLUTDumper",
    out_directory = cms.string("lut_dump"),
)

process.dumpPath = cms.Path( process.dumper )
process.schedule = cms.Schedule(process.dumpPath)
