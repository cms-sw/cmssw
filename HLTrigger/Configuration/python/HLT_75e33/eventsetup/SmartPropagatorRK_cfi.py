import FWCore.ParameterSet.Config as cms

SmartPropagatorRK = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorRK'),
    Epsilon = cms.double(5.0),
    MuonPropagator = cms.string('SteppingHelixPropagatorAlong'),
    PropagationDirection = cms.string('alongMomentum'),
    TrackerPropagator = cms.string('RungeKuttaTrackerPropagator')
)
