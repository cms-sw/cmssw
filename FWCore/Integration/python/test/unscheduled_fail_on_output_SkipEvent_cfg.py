import FWCore.ParameterSet.Config as cms

from FWCore.Integration.test.unscheduled_fail_on_output_cfg import process
process.options.SkipEvent = cms.untracked.vstring('NotFound')
