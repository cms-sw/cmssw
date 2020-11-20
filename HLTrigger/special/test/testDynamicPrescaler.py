import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(  
  input = cms.untracked.int32(10000)
)

process.load("HLTrigger.special.hltDynamicPrescaler_cfi")

process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults' )
)
process.MessageLogger.HLTrigReport=dict()

process.path = cms.Path(process.hltDynamicPrescaler)
process.info = cms.EndPath(process.hltTrigReport)
