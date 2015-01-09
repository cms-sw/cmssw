import FWCore.ParameterSet.Config as cms

# The sequences

HLTEcalActivityEgammaRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")
)
