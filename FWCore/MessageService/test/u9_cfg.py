# Unit test configuration file for MessageLogger service
# Uses include MessageLogger.cfi and rather little else

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.destinations = ['warnings', 'infos']
process.MessageLogger.statistics = ['warnings', 'infos']
process.MessageLogger.fwkJobReports = ['job_report']
process.MessageLogger.default = cms.untracked.PSet(
    noTimeStamps = cms.untracked.bool(False),
    FwkJob = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
    limit = cms.untracked.int32(-1)
)
process.MessageLogger.warnings = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(3)
    ),
    noTimeStamps = cms.untracked.bool(True)
)
process.MessageLogger.infos = cms.untracked.PSet(
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(2)
    ),
    noTimeStamps = cms.untracked.bool(True),
    FwkJob = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_G")

process.p = cms.Path(process.sendSomeMessages)
