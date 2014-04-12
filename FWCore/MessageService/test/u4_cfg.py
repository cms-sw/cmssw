# Unit test configuration file for MessageLogger service:
# statistics destination :  output name specified; 
# threshold for statistics destination

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    u4_statistics = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    statistics = cms.untracked.vstring('u4_statistics', 
        'anotherStats', 
        'u4_errors'),
    u4_errors = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        noTimeStamps = cms.untracked.bool(True),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    anotherStats = cms.untracked.PSet(
        output = cms.untracked.string('u4_another')
    ),
    categories = cms.untracked.vstring('preEventProcessing'),
    destinations = cms.untracked.vstring('u4_errors')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_A")

process.p = cms.Path(process.sendSomeMessages)
