import FWCore.ParameterSet.Config as cms

from FWCore.Integration.test.unscheduled_fail_on_output_cfg import process
process.options.IgnoreCompletely = cms.untracked.vstring('NotFound')
# foo bar baz
# EAG96aGVbFd3d
# 6o11qyMTkGAo2
