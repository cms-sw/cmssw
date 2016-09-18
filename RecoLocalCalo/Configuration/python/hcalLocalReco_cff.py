import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
hcalOOTPileupESProducer = cms.ESProducer('OOTPileupDBCompatibilityESProducer')

from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hbhe_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_zdc_cfi import *
hcalLocalRecoSequence = cms.Sequence(hbheprereco+hfreco+horeco+zdcreco)

from RecoLocalCalo.HcalRecProducers.HFUpgradeReconstructor_cfi import hfUpgradeReco as _hfUpgradeReco

_phase2_hcalLocalRecoSequence = hcalLocalRecoSequence.copy()
_phase2_hcalLocalRecoSequence.remove(hbheprereco)

from Configuration.StandardSequences.Eras import eras
eras.phase2_hcal.toModify( horeco, digiLabel = cms.InputTag('simHcalDigis') )
eras.phase2_hcal.toReplaceWith( hfreco, _hfUpgradeReco )
eras.phase2_hcal.toModify( hfreco, digiLabel = cms.InputTag('simHcalDigis','HFUpgradeDigiCollection') )
eras.phase2_hcal.toModify( zdcreco, digiLabel = cms.InputTag('simHcalUnsuppressedDigis'), digiLabelhcal = cms.InputTag('simHcalUnsuppressedDigis') )
eras.phase2_hcal.toReplaceWith( hcalLocalRecoSequence, _phase2_hcalLocalRecoSequence )

from RecoLocalCalo.HcalRecProducers.hfprereco_cfi import hfprereco
from RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi import hfreco as _phase1_hfreco
from RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi import hbheprereco as _phase1_hbheprereco

_phase1_hcalLocalRecoSequence = hcalLocalRecoSequence.copy()
_phase1_hcalLocalRecoSequence.insert(0,hfprereco)

eras.run2_HF_2017.toReplaceWith( hcalLocalRecoSequence, _phase1_hcalLocalRecoSequence )
eras.run2_HF_2017.toReplaceWith( hfreco, _phase1_hfreco )
eras.run2_HE_2017.toReplaceWith( hbheprereco, _phase1_hbheprereco )
eras.run2_HE_2017.toModify( hbheprereco, saveInfos = cms.bool(True) )
