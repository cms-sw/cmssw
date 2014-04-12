# Free test configuration file for MessageLogger service:
# Behavior implied by S. Naumann but unexpected on our part.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

import FWCore.Framework.test.cmsExceptionsFatal_cff
process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('expect_specific')
process.MessageLogger.cerr.TtSemiLeptonicEvent = cms.untracked.PSet( 
  limit = cms.untracked.int32(-1) 
)
#process.MessageLogger.cerr.INFO = cms.untracked.PSet(
#    default             = cms.untracked.PSet( limit = cms.untracked.int32( 0)
#),
#)

process.MessageLogger.cerr.INFO = cms.untracked.PSet(
   limit = cms.untracked.int32( 0),
   TtSemiLeptonicEvent = cms.untracked.PSet( 
  limit = cms.untracked.int32(-1)
  ) 
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.source = cms.Source("EmptySource")

process.sendSomeMessages = cms.EDAnalyzer( "UnitTestClient_E")

process.p = cms.Path(process.sendSomeMessages)
