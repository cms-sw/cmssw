import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBiasNoise_cff import *

import HLTrigger.HLTfilters.hltHighLevel_cfi
hcalminbiasHLT =  HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    HLTPaths = ['HLT_HcalPhiSym'],
    eventSetupPathsKey='HcalCalMinBias',
    throw = False #dont throw except on unknown path name 
)

## customizations for the pp_on_AA_2018 eras
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toModify(hcalminbiasHLT,
                       eventSetupPathsKey='HcalCalMinBiasHI'
)


import RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi
hbherecoMBNZS = RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi.hbheprereco.clone(
    digiLabelQIE8  = cms.InputTag("hcalDigiAlCaMB"),
    digiLabelQIE11 = cms.InputTag("hcalDigiAlCaMB"),
###    tsFromDB = cms.bool(False),
    dropZSmarkedPassed = cms.bool(False),
    algorithm = dict(
        useMahi = cms.bool(False),
        useM2 = cms.bool(False),
        useM3 = cms.bool(False)
    ),
    processQIE11 = cms.bool(False),
    setNegativeFlagsQIE8 = cms.bool(False),
    setNegativeFlagsQIE11 = cms.bool(False),
    setNoiseFlagsQIE8 = cms.bool(True),
    setNoiseFlagsQIE11 = cms.bool(False),
    setPulseShapeFlagsQIE8 = cms.bool(False),
    setPulseShapeFlagsQIE11 = cms.bool(False),
    setLegacyFlagsQIE8 = cms.bool(False),
    setLegacyFlagsQIE11 = cms.bool(False),
)

hbherecoMBNZS.algorithm.firstSampleShift = 0 # explicitly repeating the default

import RecoLocalCalo.HcalRecProducers.hosimplereco_cfi
horecoMBNZS = RecoLocalCalo.HcalRecProducers.hosimplereco_cfi.hosimplereco.clone()

horecoMBNZS.firstSample = 4
horecoMBNZS.samplesToAdd = 4
horecoMBNZS.digiLabel = 'hcalDigiAlCaMB'
horecoMBNZS.tsFromDB = cms.bool(False)
horecoMBNZS.dropZSmarkedPassed = cms.bool(False)

import RecoLocalCalo.HcalRecProducers.hfsimplereco_cfi
hfrecoMBNZS = RecoLocalCalo.HcalRecProducers.hfsimplereco_cfi.hfsimplereco.clone()

hfrecoMBNZS.firstSample = 2   # Run 2 default before 2017
hfrecoMBNZS.samplesToAdd = 2
hfrecoMBNZS.digiLabel = 'hcalDigiAlCaMB'
hfrecoMBNZS.tsFromDB = cms.bool(False)
hfrecoMBNZS.dropZSmarkedPassed = cms.bool(False)

seqALCARECOHcalCalMinBiasDigi = cms.Sequence(hcalminbiasHLT*hcalDigiAlCaMB*gtDigisAlCaMB)
seqALCARECOHcalCalMinBiasDigiNoHLT = cms.Sequence(hcalDigiAlCaMB*gtDigisAlCaMB)

seqALCARECOHcalCalMinBias = cms.Sequence(hbherecoMBNZS*horecoMBNZS*hbherecoNoise*hfrecoNoise*hfrecoMBNZS*horecoNoise)

#Specify to use HI output for the pp_on_AA_2018 eras
seqALCARECOHcalCalMinBiasHI = cms.Sequence(hbherecoNoise*hfrecoNoise*hfrecoMBNZS*horecoNoise)
pp_on_AA_2018.toReplaceWith(seqALCARECOHcalCalMinBias,
                            seqALCARECOHcalCalMinBiasHI
)

import RecoLocalCalo.HcalRecProducers.hfprereco_cfi
hfprerecoNoise = RecoLocalCalo.HcalRecProducers.hfprereco_cfi.hfprereco.clone(
    digiLabel = cms.InputTag("hcalDigiAlCaMB"),
    dropZSmarkedPassed = cms.bool(False),
    tsFromDB = cms.bool(False),
    sumAllTimeSlices = cms.bool(False),
    forceSOI = cms.int32(0)
)
hfprerecoMBNZS = RecoLocalCalo.HcalRecProducers.hfprereco_cfi.hfprereco.clone(
    digiLabel = cms.InputTag("hcalDigiAlCaMB"),
    dropZSmarkedPassed = cms.bool(False),
    tsFromDB = cms.bool(False),
    sumAllTimeSlices = cms.bool(True),
    forceSOI = cms.int32(1)
)

import RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi
_phase1_hfrecoNoise = RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi.hfreco.clone(
    inputLabel = cms.InputTag("hfprerecoNoise"),
    setNoiseFlags = cms.bool(False),
    algorithm = dict(
        Class = cms.string("HFSimpleTimeCheck"),
        rejectAllFailures = cms.bool(False)
    ),
)
_phase1_hfrecoMBNZS = RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi.hfreco.clone(
    inputLabel = cms.InputTag("hfprerecoMBNZS"),
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
hbheplan1MBNZS = RecoLocalCalo.HcalRecProducers.hbheplan1_cfi.hbheplan1.clone(
    hbheInput = cms.InputTag("hbheprerecoMBNZS")
)
hbheplan1Noise = RecoLocalCalo.HcalRecProducers.hbheplan1_cfi.hbheplan1.clone(
    hbheInput = cms.InputTag("hbheprerecoNoise")
)

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( hbherecoMBNZS,
    processQIE11 = cms.bool(True),
# temporarily disabled until RecoLocalCalo/HcalRecProducers/python/HBHEPhase1Reconstructor_cfi.py:flagParametersQIE11 is filled
#    setNoiseFlagsQIE11 = cms.bool(True),
)
run2_HCAL_2017.toModify( hbherecoNoise,
    processQIE11 = cms.bool(True),
# temporarily disabled until RecoLocalCalo/HcalRecProducers/python/HBHEPhase1Reconstructor_cfi.py:flagParametersQIE11 is filled
#    setNoiseFlagsQIE11 = cms.bool(True),
)

_plan1_seqALCARECOHcalCalMinBias = _phase1_seqALCARECOHcalCalMinBias.copy()
hbheprerecoMBNZS = hbherecoMBNZS.clone()
hbheprerecoNoise = hbherecoNoise.clone()
_plan1_seqALCARECOHcalCalMinBias.insert(0,hbheprerecoNoise)
_plan1_seqALCARECOHcalCalMinBias.insert(0,hbheprerecoMBNZS)
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toReplaceWith(hbherecoMBNZS, hbheplan1MBNZS)
run2_HEPlan1_2017.toReplaceWith(hbherecoNoise, hbheplan1Noise)
run2_HEPlan1_2017.toReplaceWith(seqALCARECOHcalCalMinBias, _plan1_seqALCARECOHcalCalMinBias)
