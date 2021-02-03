import FWCore.ParameterSet.Config as cms

requireLeadPion = cms.PSet(
    BooleanOperator = cms.string('and'),
    leadPion = cms.PSet(
        Producer = cms.InputTag("pfRecoTauDiscriminationByLeadingTrackFinding"),
        cut = cms.double(0.5)
    )
)