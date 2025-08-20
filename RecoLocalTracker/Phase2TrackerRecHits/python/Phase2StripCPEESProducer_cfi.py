import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.Phase2TrackerRecHits.phase2StripCPEESProducer_cfi import phase2StripCPEESProducer as _phase2StripCPEESProducer
phase2StripCPEESProducer = _phase2StripCPEESProducer.clone(ComponentType = 'Phase2StripCPE',
                                                           parameters    = dict(LorentzAngle_DB = True,
                                                                                TanLorentzAnglePerTesla = 0.07))
