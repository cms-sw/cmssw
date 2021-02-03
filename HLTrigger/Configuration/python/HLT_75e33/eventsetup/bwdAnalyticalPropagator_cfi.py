import FWCore.ParameterSet.Config as cms

bwdAnalyticalPropagator = cms.ESProducer("AnalyticalPropagatorESProducer",
    ComponentName = cms.string('bwdAnalyticalPropagator'),
    MaxDPhi = cms.double(1.6),
    PropagationDirection = cms.string('oppositeToMomentum')
)
