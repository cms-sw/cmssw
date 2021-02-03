import FWCore.ParameterSet.Config as cms

SmartPropagator = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagator'),
    Epsilon = cms.double(5.0),
    MuonPropagator = cms.string('SteppingHelixPropagatorAlong'),
    PropagationDirection = cms.string('alongMomentum'),
    TrackerPropagator = cms.string('PropagatorWithMaterial')
)
