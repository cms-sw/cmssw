# Unit test configuration file for MessageLogger service:
# distinct threshold level for linked destination.
# A destination with different PSet name uses output= to share the
# same stream as an ordinary output destination.  But their thresholds
# are different.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
      u28_output = cms.untracked.PSet(
          threshold = cms.untracked.string('INFO'),
          noTimeStamps = cms.untracked.bool(True),
          enableStatistics = cms.untracked.bool(True),
          statisticsThreshold = cms.untracked.string('WARNING')
      )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)
