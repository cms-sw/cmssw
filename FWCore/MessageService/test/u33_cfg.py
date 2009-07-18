# Unit test configuration file for non-per-event module logging
#   Tests that each type of activity gets right module label
#   Tests for suppression and enabling accordingly

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.MessageLogger = cms.Service("MessageLogger",
   destinations = cms.untracked.vstring('u33_all'),
    statistics = cms.untracked.vstring('u33_all'),
    categories = cms.untracked.vstring('cat_A'), 
    suppressInfo = cms.untracked.vstring('ssm_2a'), 
    debugModules = cms.untracked.vstring('ssm_1b'), 
    u33_all = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        noTimeStamps = cms.untracked.bool(True),
        default = cms.untracked.PSet(
                    limit = cms.untracked.int32(-1)
        ),
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.ssm_1a = cms.EDAnalyzer("UTC_V1",
    identifier = cms.untracked.int32(11)
)

process.ssm_1b = cms.EDAnalyzer("UTC_V1",
    identifier = cms.untracked.int32(12)
)

process.ssm_2a = cms.EDAnalyzer("UTC_V2",
    identifier = cms.untracked.int32(21)
)

process.ssm_2b = cms.EDAnalyzer("UTC_V2",
    identifier = cms.untracked.int32(22)
)

process.q = cms.Path(process.ssm_1a*process.ssm_2a*process.ssm_1b*process.ssm_2b)
