import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *
hcalGlobalRecoSequence = cms.Sequence(hbhereco)

from RecoLocalCalo.HcalRecProducers.HBHEUpgradeReconstructor_cfi import hbheUpgradeReco as _hbheUpgradeReco

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toReplaceWith( hbhereco, _hbheUpgradeReco )
phase2_hcal.toModify( hbhereco, digiLabel = cms.InputTag('simHcalDigis','HBHEUpgradeDigiCollection') )
