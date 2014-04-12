# Test that the ParameterSetDescription validation
# is being run by defining an illegal parameter.
# An exception should be thrown and the job stops.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr_stats.output = 'NULL'

process.load("FWCore.Modules.printContent_cfi")
# Intentionally define a parameter that does not
# exist in the ParameterSetDescription which
# should result in an exception.
process.printContent.doesNotExist = cms.int32(1)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.p = cms.Path(process.printContent)
