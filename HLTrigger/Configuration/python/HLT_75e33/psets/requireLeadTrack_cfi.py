import FWCore.ParameterSet.Config as cms

requireLeadTrack = cms.PSet(
    BooleanOperator = cms.string('and'),
    leadTrack = cms.PSet(
        Producer = cms.InputTag("pfRecoTauDiscriminationByLeadingTrackFinding"),
        cut = cms.double(0.5)
    )
)