import FWCore.ParameterSet.Config as cms

hltSmartPropagatorAnyRK = cms.ESProducer("SmartPropagatorESProducer",
                                         ComponentName = cms.string('hltSmartPropagatorAnyRK'),
                                         TrackerPropagator = cms.string('RungeKuttaTrackerPropagator'),
                                         MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
                                         PropagationDirection = cms.string('alongMomentum'),
                                         Epsilon = cms.double(5.0)
                                         )
