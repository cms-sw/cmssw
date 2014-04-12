import FWCore.ParameterSet.Config as cms

apvshotsfilter = cms.EDFilter('APVShotsFilter',
     digiCollection     = cms.InputTag("siStripDigis","ZeroSuppressed"),
     historyProduct     = cms.InputTag("consecutiveHEs"),
     apvPhaseCollection = cms.InputTag("APVPhases"),
     zeroSuppressed     = cms.untracked.bool(True),
     useCabling         = cms.untracked.bool(False),
     selectAPVshots     = cms.untracked.bool(True)
)	
