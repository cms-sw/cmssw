# Unit test configuration file for FlushMessageLog in MessageLogger service:
#   Tests effect of LogFlush by cfg-configurable choices of how many 
#   messages to use to clog the queue and whether or not FlushMessageLog
#   is invoked.  Under normal testing, it will invoke FlushMessageLog in
#   a situation where its absence would result in different output.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.MessageLogger = cms.Service("MessageLogger",
    default = cms.untracked.PSet(
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    u22_warnings = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noTimeStamps = cms.untracked.bool(True),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('preEventProcessing', 
        'FwkJob'),
    destinations = cms.untracked.vstring('u22_warnings')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_P",
    queueFillers = cms.untracked.int32(500),
    useLogFlush = cms.untracked.bool(True)
)

process.p = cms.Path(process.sendSomeMessages)
