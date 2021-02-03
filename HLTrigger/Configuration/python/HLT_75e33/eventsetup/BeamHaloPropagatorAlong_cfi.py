import FWCore.ParameterSet.Config as cms

BeamHaloPropagatorAlong = cms.ESProducer("BeamHaloPropagatorESProducer",
    ComponentName = cms.string('BeamHaloPropagatorAlong'),
    CrossingTrackerPropagator = cms.string('BeamHaloSHPropagatorAlong'),
    EndCapTrackerPropagator = cms.string('BeamHaloMPropagatorAlong'),
    PropagationDirection = cms.string('alongMomentum')
)
