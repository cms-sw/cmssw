import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBiasNoise_cff import *

import HLTrigger.HLTfilters.hltHighLevel_cfi
hcalminbiasHLT =  HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    HLTPaths = ['HLT_HcalPhiSym'],
    eventSetupPathsKey='HcalCalMinBiasHI',
    throw = False #dont throw except on unknown path name 
)

import RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi
hfrecoMBNZS = RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi.hfreco.clone()

hfrecoMBNZS.firstSample = 2
hfrecoMBNZS.samplesToAdd = 2
hfrecoMBNZS.digiLabel = 'hcalDigiAlCaMB'
hfrecoMBNZS.tsFromDB = cms.bool(False)
hfrecoMBNZS.dropZSmarkedPassed = cms.bool(False)

import RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi

seqALCARECOHcalCalMinBiasDigi = cms.Sequence(hcalminbiasHLT*hcalDigiAlCaMB*gtDigisAlCaMB)
seqALCARECOHcalCalMinBias = cms.Sequence(hbherecoNoise*hfrecoNoise*hfrecoMBNZS*horecoNoise)

import RecoLocalCalo.HcalRecProducers.hfprereco_cfi
hfprerecoMBNZS = RecoLocalCalo.HcalRecProducers.hfprereco_cfi.hfprereco.clone(
    digiLabel = cms.InputTag("hcalDigiAlCaMB"),
    dropZSmarkedPassed = cms.bool(False),
    tsFromDB = cms.bool(False),
    sumAllTimeSlices = cms.bool(True),
    forceSOI = cms.int32(1)
)
hfprerecoNoise = RecoLocalCalo.HcalRecProducers.hfprereco_cfi.hfprereco.clone(
    digiLabel = cms.InputTag("hcalDigiAlCaMB"),
    dropZSmarkedPassed = cms.bool(False),
    tsFromDB = cms.bool(False),
    sumAllTimeSlices = cms.bool(False),
    forceSOI = cms.int32(0)
)

import RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi
_phase1_hfrecoMBNZS = RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi.hfreco.clone(
    inputLabel = cms.InputTag("hfprerecoMBNZS"),
    setNoiseFlags = cms.bool(False),
    algorithm = dict(
        Class = cms.string("HFSimpleTimeCheck"),
        rejectAllFailures = cms.bool(False)
    ),
)
_phase1_hfrecoNoise = RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi.hfreco.clone(
    inputLabel = cms.InputTag("hfprerecoNoise"),
    setNoiseFlags = cms.bool(False),
    algorithm = dict(
        Class = cms.string("HFSimpleTimeCheck"),
        rejectAllFailures = cms.bool(False)
    ),
)

_phase1_seqALCARECOHcalCalMinBias = seqALCARECOHcalCalMinBias.copy()
_phase1_seqALCARECOHcalCalMinBias.insert(0,hfprerecoMBNZS)
_phase1_seqALCARECOHcalCalMinBias.insert(0,hfprerecoNoise)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toReplaceWith( seqALCARECOHcalCalMinBias, _phase1_seqALCARECOHcalCalMinBias )
run2_HF_2017.toReplaceWith( hfrecoNoise, _phase1_hfrecoNoise )
run2_HF_2017.toReplaceWith( hfrecoMBNZS, _phase1_hfrecoMBNZS )

import RecoLocalCalo.HcalRecProducers.hbheplan1_cfi
hbheplan1Noise = RecoLocalCalo.HcalRecProducers.hbheplan1_cfi.hbheplan1.clone(
    hbheInput = cms.InputTag("hbheprerecoNoise")
)

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( hbherecoNoise,
    processQIE11 = cms.bool(True),
# temporarily disabled until RecoLocalCalo/HcalRecProducers/python/HBHEPhase1Reconstructor_cfi.py:flagParametersQIE11 is filled
#    setNoiseFlagsQIE11 = cms.bool(True),
)

_plan1_seqALCARECOHcalCalMinBias = _phase1_seqALCARECOHcalCalMinBias.copy()
hbheprerecoNoise = hbherecoNoise.clone()
_plan1_seqALCARECOHcalCalMinBias.insert(0,hbheprerecoNoise)
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toReplaceWith(hbherecoNoise, hbheplan1Noise)
run2_HEPlan1_2017.toReplaceWith(seqALCARECOHcalCalMinBias, _plan1_seqALCARECOHcalCalMinBias)
