import FWCore.ParameterSet.Config as cms

SmartPropagatorRKOpposite = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorRKOpposite'),
    Epsilon = cms.double(5.0),
    MuonPropagator = cms.string('SteppingHelixPropagatorOpposite'),
    PropagationDirection = cms.string('oppositeToMomentum'),
    TrackerPropagator = cms.string('RungeKuttaTrackerPropagatorOpposite')
)
