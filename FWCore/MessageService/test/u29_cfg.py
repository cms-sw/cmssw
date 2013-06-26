# Unit test configuration file for MessageLogger service
# Tests the hardwired defaults
# Does not include MessageLogger.cfi nor explicitly mention MessageLogger
# or MessageService at all.
# Not suitable for unit test because the time stamps will not be disabled

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

#process.load("FWCore.MessageService.test.Services_cff")

#process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.default = cms.untracked.PSet(
#    noTimeStamps = cms.untracked.bool(True)
#)
#process.MessageLogger.cerr.noTimeStamps = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)
