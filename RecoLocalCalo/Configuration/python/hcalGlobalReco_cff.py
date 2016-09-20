import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *
hcalGlobalRecoSequence = cms.Sequence(hbhereco)

from RecoLocalCalo.HcalRecProducers.HBHEUpgradeReconstructor_cfi import hbheUpgradeReco as _hbheUpgradeReco

from Configuration.StandardSequences.Eras import eras
eras.phase2_hcal.toReplaceWith( hbhereco, _hbheUpgradeReco )
eras.phase2_hcal.toModify( hbhereco, digiLabel = cms.InputTag('simHcalDigis','HBHEUpgradeDigiCollection') )
