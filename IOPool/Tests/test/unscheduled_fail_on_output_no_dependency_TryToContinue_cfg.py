import FWCore.ParameterSet.Config as cms

from FWCore.Integration.test.unscheduled_fail_on_output_cfg import process
process.options.TryToContinue = ['NotFound']

process.out.outputCommands = cms.untracked.vstring('keep *', 'drop *_failing_*_*')
process.failGet = cms.EDAnalyzer('IntTestAnalyzer', moduleLabel = cms.untracked.InputTag('failing'), valueMustMatch = cms.untracked.int32(0))
process.failingEnd = cms.EndPath(process.failGet)
