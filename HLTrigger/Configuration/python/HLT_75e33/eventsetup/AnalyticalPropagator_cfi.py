import FWCore.ParameterSet.Config as cms

AnalyticalPropagator = cms.ESProducer("AnalyticalPropagatorESProducer",
    ComponentName = cms.string('AnalyticalPropagator'),
    MaxDPhi = cms.double(1.6),
    PropagationDirection = cms.string('alongMomentum')
)
