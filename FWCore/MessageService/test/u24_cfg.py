# Unit test configuration file for MessageLogger service:
# Behavior when duplicate file names are supplied.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    default = cms.untracked.PSet(
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    # produce SAME file u24.log via warnings config - should be ignored!
    u24_warnings = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noTimeStamps = cms.untracked.bool(True),
        extension = cms.untracked.string('log'),
        filename = cms.untracked.string('u24')
    ),
    debugModules = cms.untracked.vstring('*'),
    # produce file u24.log
    u24_errors = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        noTimeStamps = cms.untracked.bool(True),
        extension = cms.untracked.string('log'),
        filename = cms.untracked.string('u24')
    ),
    categories = cms.untracked.vstring('preEventProcessing', 
        'FwkJob'),
    destinations = cms.untracked.vstring('u24_warnings', 
        'u24_errors')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)
