# Unit test configuration file for MessageLogger service:
# Suppression based on minimum severity level.
# (This is part A - verify that the suppression occurs.
# u35 is part B - verifying that suppression does not occur if any threshold
# is low enough to not suppress.)
#

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    u34_warnings = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        noTimeStamps = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('u34_warnings',
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_W")

process.p = cms.Path(process.sendSomeMessages)
