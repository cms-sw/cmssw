# Unit test configuration file for MessageLogger service:
#   Filtering all but one category (for example FwkTest)

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    u7_restrict = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(True),
        special = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    u7_log = cms.untracked.PSet(
        noTimeStamps = cms.untracked.bool(True),
        FwkTest = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('u7_log', 
        'u7_restrict'),
    categories = cms.untracked.vstring('FwkTest', 
        'special')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_D")

process.p = cms.Path(process.sendSomeMessages)


