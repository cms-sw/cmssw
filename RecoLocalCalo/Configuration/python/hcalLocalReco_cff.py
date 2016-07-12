import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
hcalOOTPileupESProducer = cms.ESProducer('OOTPileupDBCompatibilityESProducer')

from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_zdc_cfi import *
hcalLocalRecoSequence = cms.Sequence(hbheprereco+hfreco+horeco+zdcreco)

#from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *
#hcalGlobalRecoSequence = cms.Sequence(hbhereco)

from RecoLocalCalo.HcalRecProducers.HBHEUpgradeReconstructor_cfi import *
from RecoLocalCalo.HcalRecProducers.HFUpgradeReconstructor_cfi import *

_phase2_hcalLocalRecoSequence = hcalLocalRecoSequence.copy()
_phase2_hcalLocalRecoSequence.replace(hfreco,hfUpgradeReco)
_phase2_hcalLocalRecoSequence.replace(hbheprereco,hbheUpgradeReco)

from Configuration.StandardSequences.Eras import eras
eras.phase2_hcal.toModify( hbheUpgradeReco, digiLabel = cms.InputTag('simHcalDigis','HBHEUpgradeDigiCollection') )
eras.phase2_hcal.toModify( horeco, digiLabel = cms.InputTag('simHcalDigis') )
eras.phase2_hcal.toModify( hfUpgradeReco, digiLabel = cms.InputTag('simHcalDigis','HFUpgradeDigiCollection') )
eras.phase2_hcal.toModify( zdcreco, digiLabel = cms.InputTag('simHcalUnsuppressedDigis'), digiLabelhcal = cms.InputTag('simHcalUnsuppressedDigis') )
eras.phase2_hcal.toReplaceWith( hcalLocalRecoSequence, _phase2_hcalLocalRecoSequence )
