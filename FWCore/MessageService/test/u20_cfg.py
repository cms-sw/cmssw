# Unit test configuration file for MessageLogger service
# Uses include MessageLogger.cfi and nothing else except time stamp suppression
# Currently output will be jumbled unless cout and cerr are directed separately

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.default = cms.untracked.PSet(
    noTimeStamps = cms.untracked.bool(True)
)
process.MessageLogger.cerr.noTimeStamps = True
process.MessageLogger.cerr.enableStatistics = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer("UnitTestClient_G")

process.p = cms.Path(process.sendSomeMessages)
