import FWCore.ParameterSet.Config as cms

StraightLinePropagator = cms.ESProducer("StraightLinePropagatorESProducer",
    ComponentName = cms.string('StraightLinePropagator'),
    PropagationDirection = cms.string('alongMomentum')
)
