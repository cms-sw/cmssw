# Unit test configuration file for MessageLogger service:
# statistics destination :  output name specified; 
# threshold for statistics destination

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
        u4_statistics = cms.untracked.PSet(
            threshold = cms.untracked.string('WARNING'),
            default = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
            ),
            enableStatistics = cms.untracked.bool(True)
        ),
        u4_errors = cms.untracked.PSet(
            threshold = cms.untracked.string('ERROR'),
            noTimeStamps = cms.untracked.bool(True),
            enableStatistics = cms.untracked.bool(True)
        ),
        anotherStats = cms.untracked.PSet(
            output = cms.untracked.string('u4_another'),
            enableStatistics = cms.untracked.bool(True),
            default = cms.untracked.PSet(
              limit = cms.untracked.int32(0)
            )
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)
