# Free test configuration file for MessageLogger service:
# Behavior implied by S. Naumann but unexpected on our part.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer( "PSetTestClient_A", 
  a = cms.untracked.PSet 
  (
    b = cms.untracked.PSet
    (
      x = cms.untracked.int32(4)
    )
  )
)

process.p = cms.Path(process.sendSomeMessages)
