import FWCore.ParameterSet.Config as cms

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi
hbherecoMB = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi.hbheprereco.clone()

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoMB = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi
horecoMB = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi.horeco.clone()


# switch off "Hcal ZS in reco":
hbherecoMB.dropZSmarkedPassed = cms.bool(False)
hfrecoMB.dropZSmarkedPassed = cms.bool(False)
horecoMB.dropZSmarkedPassed = cms.bool(False)

hcalLocalRecoSequenceNZS = cms.Sequence(hbherecoMB*hfrecoMB*horecoMB) 

import RecoLocalCalo.HcalRecProducers.hfprereco_cfi
import RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi
import RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi
_phase1_hbherecoMB = RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi.hbheprereco.clone(
    dropZSmarkedPassed = cms.bool(False),
    recoParamsFromDB = cms.bool(False),
    algorithm = dict(
        useM2 = cms.bool(False),
        useM3 = cms.bool(False)
    ),
)
hfprerecoMB = RecoLocalCalo.HcalRecProducers.hfprereco_cfi.hfprereco.clone(
    dropZSmarkedPassed = cms.bool(False)
)
_phase1_hfrecoMB = RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi.hfreco.clone(
    inputLabel = cms.InputTag("hfprerecoMB"),
    setNoiseFlags = cms.bool(False),
    algorithm = dict(
        Class = cms.string("HFSimpleTimeCheck"),
        rejectAllFailures = cms.bool(False)
    ),
)
import RecoLocalCalo.HcalRecProducers.hbheplan1_cfi
hbheplan1MB = RecoLocalCalo.HcalRecProducers.hbheplan1_cfi.hbheplan1.clone(
    hbheInput = cms.InputTag("hbheprerecoMB")
)

_phase1_hcalLocalRecoSequenceNZS = hcalLocalRecoSequenceNZS.copy()
_phase1_hcalLocalRecoSequenceNZS.insert(0,hfprerecoMB)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toReplaceWith( hcalLocalRecoSequenceNZS, _phase1_hcalLocalRecoSequenceNZS )
run2_HF_2017.toReplaceWith( hfrecoMB, _phase1_hfrecoMB )
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toReplaceWith( hbherecoMB, _phase1_hbherecoMB )

_plan1_hcalLocalRecoSequenceNZS = _phase1_hcalLocalRecoSequenceNZS.copy()
hbheprerecoMB = _phase1_hbherecoMB.clone()
_plan1_hcalLocalRecoSequenceNZS.insert(0,hbheprerecoMB)
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toReplaceWith(hbherecoMB, hbheplan1MB)
run2_HEPlan1_2017.toReplaceWith(hcalLocalRecoSequenceNZS, _plan1_hcalLocalRecoSequenceNZS)
