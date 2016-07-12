import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *
hcalGlobalRecoSequence = cms.Sequence(hbhereco)

from Configuration.StandardSequences.Eras import eras
eras.phase2_hcal.toReplaceWith( hcalGlobalRecoSequence, cms.Sequence() )
