import FWCore.ParameterSet.Config as cms

from unscheduled_fail_on_output_cfg import process
process.options.Rethrow = cms.untracked.vstring('NotFound')
