import FWCore.ParameterSet.Config as cms

def _addProcessSmartPropagatorAnyRK(process):
    process.hltSmartPropagatorAnyRK = cms.ESProducer("SmartPropagatorESProducer",
                                                     ComponentName = cms.string('hltSmartPropagatorAnyRK'),
                                                     TrackerPropagator = cms.string('RungeKuttaTrackerPropagator'),
                                                     MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
                                                     PropagationDirection = cms.string('alongMomentum'),
                                                     Epsilon = cms.double(5.0))

from Configuration.ProcessModifiers.mtd_at_hlt_cff import mtd_at_hlt
modifyConfigurationForSmartPropagatorAnyRK_ = mtd_at_hlt.makeProcessModifier(_addProcessSmartPropagatorAnyRK)
