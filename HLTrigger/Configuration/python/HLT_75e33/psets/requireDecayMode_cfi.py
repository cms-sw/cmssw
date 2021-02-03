import FWCore.ParameterSet.Config as cms

requireDecayMode = cms.PSet(
    BooleanOperator = cms.string('and'),
    decayMode = cms.PSet(
        Producer = cms.InputTag("hpsPFTauDiscriminationByDecayModeFindingNewDMs"),
        cut = cms.double(0.5)
    )
)