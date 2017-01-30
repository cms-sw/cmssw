import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
hcalOOTPileupESProducer = cms.ESProducer('OOTPileupDBCompatibilityESProducer')

from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_zdc_cfi import *
hcalLocalRecoSequence = cms.Sequence(hbheprereco+hfreco+horeco+zdcreco)

from RecoLocalCalo.HcalRecProducers.hfprereco_cfi import hfprereco
from RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi import hfreco as _phase1_hfreco
from RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi import hbheprereco as _phase1_hbheprereco

# copies for cosmics
_default_hbheprereco = hbheprereco.clone()
_default_hfreco = hfreco.clone()

_phase1_hcalLocalRecoSequence = hcalLocalRecoSequence.copy()
_phase1_hcalLocalRecoSequence.insert(0,hfprereco)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toReplaceWith( hcalLocalRecoSequence, _phase1_hcalLocalRecoSequence )
run2_HF_2017.toReplaceWith( hfreco, _phase1_hfreco )
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toReplaceWith( hbheprereco, _phase1_hbheprereco )

_phase2_hcalLocalRecoSequence = hcalLocalRecoSequence.copy()
_phase2_hcalLocalRecoSequence.remove(hbheprereco)

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify( horeco, digiLabel = cms.InputTag('simHcalDigis') )
phase2_hcal.toModify( hfprereco, digiLabel = cms.InputTag('simHcalDigis','HFQIE10DigiCollection') )
phase2_hcal.toModify( zdcreco, digiLabel = cms.InputTag('simHcalUnsuppressedDigis'), digiLabelhcal = cms.InputTag('simHcalUnsuppressedDigis') )
phase2_hcal.toReplaceWith( hcalLocalRecoSequence, _phase2_hcalLocalRecoSequence )

