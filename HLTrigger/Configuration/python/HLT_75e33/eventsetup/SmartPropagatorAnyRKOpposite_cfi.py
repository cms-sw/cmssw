import FWCore.ParameterSet.Config as cms

SmartPropagatorAnyRKOpposite = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorAnyRKOpposite'),
    Epsilon = cms.double(5.0),
    MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
    PropagationDirection = cms.string('oppositeToMomentum'),
    TrackerPropagator = cms.string('RungeKuttaTrackerPropagatorOpposite')
)
