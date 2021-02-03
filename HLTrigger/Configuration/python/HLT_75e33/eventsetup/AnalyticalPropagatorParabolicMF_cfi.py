import FWCore.ParameterSet.Config as cms

AnalyticalPropagatorParabolicMF = cms.ESProducer("AnalyticalPropagatorESProducer",
    ComponentName = cms.string('AnalyticalPropagatorParabolicMf'),
    MaxDPhi = cms.double(1.6),
    PropagationDirection = cms.string('alongMomentum'),
    SimpleMagneticField = cms.string('ParabolicMf')
)
