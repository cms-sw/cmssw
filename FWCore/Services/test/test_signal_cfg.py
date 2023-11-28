import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 1

process.signal = cms.EDAnalyzer("SignallingAnalyzer", signal = cms.untracked.string("INT"))

process.p = cms.Path(process.signal)
