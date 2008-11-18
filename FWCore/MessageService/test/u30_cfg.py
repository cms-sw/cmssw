# Unit test configuration file for GroupLogStatistics(category)
#   Tests effect of GroupLogStatistics(category) 
#   by having 6 differently labeled modules (of two distinct classes)
#   all of which issue LogIssue with two "grouped" categories.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.MessageLogger = cms.Service("MessageLogger",
    default = cms.untracked.PSet(
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(1000)
        )
    ),
    u30_infos = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
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
    destinations = cms.untracked.vstring('u30_infos')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

process.source = cms.Source("EmptySource")

process.ssm_1a = cms.EDAnalyzer("UTC_S1",
    identifier = cms.untracked.int32(11)
)


process.ssm_2a = cms.EDAnalyzer("UTC_S2",
    identifier = cms.untracked.int32(21)
)


process.ssm_sum = cms.EDAnalyzer("UTC_SUMMARY"
)

process.p = cms.Path(process.ssm_1a*process.ssm_2a*process.ssm_sum)
