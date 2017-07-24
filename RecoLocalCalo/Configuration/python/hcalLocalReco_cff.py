import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
hcalOOTPileupESProducer = cms.ESProducer('OOTPileupDBCompatibilityESProducer')

from RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi import hbheprereco as _phase1_hbheprereco
hbheprereco = _phase1_hbheprereco.clone(
    processQIE11 = cms.bool(False),
    tsFromDB = cms.bool(True),
    pulseShapeParametersQIE8 = dict(
        TrianglePeakTS = cms.uint32(4),
    )
)

from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_ho_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_hf_cfi import *
from RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_zdc_cfi import *
hcalLocalRecoSequence = cms.Sequence(hbheprereco+hfreco+horeco+zdcreco)

from RecoLocalCalo.HcalRecProducers.hfprereco_cfi import hfprereco
from RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi import hfreco as _phase1_hfreco
from RecoLocalCalo.HcalRecProducers.hbheplan1_cfi import hbheplan1

# copy for cosmics
_default_hfreco = hfreco.clone()

_phase1_hcalLocalRecoSequence = hcalLocalRecoSequence.copy()
_phase1_hcalLocalRecoSequence.insert(0,hfprereco)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toReplaceWith( hcalLocalRecoSequence, _phase1_hcalLocalRecoSequence )
run2_HF_2017.toReplaceWith( hfreco, _phase1_hfreco )
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toReplaceWith( hbheprereco, _phase1_hbheprereco )

_plan1_hcalLocalRecoSequence = _phase1_hcalLocalRecoSequence.copy()
_plan1_hcalLocalRecoSequence += hbheplan1
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toReplaceWith(hcalLocalRecoSequence, _plan1_hcalLocalRecoSequence)

_phase2_hcalLocalRecoSequence = hcalLocalRecoSequence.copy()
_phase2_hcalLocalRecoSequence.remove(hbheprereco)

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toReplaceWith( hcalLocalRecoSequence, _phase2_hcalLocalRecoSequence )

