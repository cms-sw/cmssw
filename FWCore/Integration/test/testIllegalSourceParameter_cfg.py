# Test that the ParameterSetDescription validation
# is being run by defining an illegal parameter.
# An exception should be thrown and the job stops.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr_stats.output = 'NULL'

# Intentionally define a parameter that does not
# exist in the ParameterSetDescription which
# should result in an exception.
process.source = cms.Source("PoolSource",
                            doesNotExist = cms.bool(True))

