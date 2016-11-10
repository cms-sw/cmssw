import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *
hcalGlobalRecoSequence = cms.Sequence(hbhereco)

from RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi import hbheprereco as _phase1_hbheprereco

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toReplaceWith( hbhereco, _phase1_hbheprereco )
phase2_hcal.toModify( hbhereco,
    digiLabelQIE8 = cms.InputTag('simHcalDigis'),
    digiLabelQIE11 = cms.InputTag('simHcalDigis','HBHEQIE11DigiCollection')
)
