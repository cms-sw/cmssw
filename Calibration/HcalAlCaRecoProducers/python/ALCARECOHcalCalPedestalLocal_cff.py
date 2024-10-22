import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for HCAL minbias:
#------------------------------------------------

from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBiasNoise_cff import *
hcalDigiAlCaMB.InputLabel = 'source'

import HLTrigger.HLTfilters.hltHighLevel_cfi
hcalminbiasHLT =  HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
#    HLTPaths = ['HLT_HcalPhiSym'],
    eventSetupPathsKey='HcalCalMinBias',
    throw = False #dont throw except on unknown path name 
)

import RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi
hbherecoMBNZS = RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi.hbheprereco.clone(
    digiLabelQIE8  = "hcalDigiAlCaMB",
    digiLabelQIE11 = "hcalDigiAlCaMB",
###    tsFromDB = False,
    dropZSmarkedPassed = False,
    algorithm = dict(
        useMahi = False,
        useM2 = False,
        useM3 = False
    ),
    processQIE11 = False,
    setNegativeFlagsQIE8 = False,
    setNegativeFlagsQIE11 = False,
    setNoiseFlagsQIE8 = True,
    setNoiseFlagsQIE11 = False,
    setPulseShapeFlagsQIE8 = False,
    setPulseShapeFlagsQIE11 = False,
    setLegacyFlagsQIE8 = False,
    setLegacyFlagsQIE11 = False,
)

hbherecoMBNZS.algorithm.firstSampleShift = 0 # explicitly repeating the default

import RecoLocalCalo.HcalRecProducers.hosimplereco_cfi
horecoMBNZS = RecoLocalCalo.HcalRecProducers.hosimplereco_cfi.hosimplereco.clone()

horecoMBNZS.firstSample = 4
horecoMBNZS.samplesToAdd = 4
horecoMBNZS.digiLabel = 'hcalDigiAlCaMB'
horecoMBNZS.tsFromDB = False
horecoMBNZS.dropZSmarkedPassed = False

import RecoLocalCalo.HcalRecProducers.hfsimplereco_cfi
hfrecoMBNZS = RecoLocalCalo.HcalRecProducers.hfsimplereco_cfi.hfsimplereco.clone()

hfrecoMBNZS.firstSample = 2   # Run 2 default before 2017
hfrecoMBNZS.samplesToAdd = 2
hfrecoMBNZS.digiLabel = 'hcalDigiAlCaMB'
hfrecoMBNZS.tsFromDB = False
hfrecoMBNZS.dropZSmarkedPassed = False

seqALCARECOHcalCalMinBiasDigi = cms.Sequence(hcalminbiasHLT*hcalDigiAlCaMB*gtDigisAlCaMB)
seqALCARECOHcalCalMinBiasDigiNoHLT = cms.Sequence(hcalDigiAlCaMB*gtDigisAlCaMB)

seqALCARECOHcalCalMinBias = cms.Sequence(hbherecoMBNZS*horecoMBNZS*hbherecoNoise*hfrecoNoise*hfrecoMBNZS*horecoNoise)
#seqALCARECOHcalCalMinBias = cms.Sequence(hbherecoMBNZS*hbherecoNoise*hfrecoNoise*hfrecoMBNZS)

import RecoLocalCalo.HcalRecProducers.hfprereco_cfi
hfprerecoNoise = RecoLocalCalo.HcalRecProducers.hfprereco_cfi.hfprereco.clone(
    digiLabel = "hcalDigiAlCaMB",
    dropZSmarkedPassed = False,
    tsFromDB = False,
    sumAllTimeSlices = False,
    forceSOI = 0
)
hfprerecoMBNZS = RecoLocalCalo.HcalRecProducers.hfprereco_cfi.hfprereco.clone(
    digiLabel = "hcalDigiAlCaMB",
    dropZSmarkedPassed = False,
    tsFromDB = False,
    sumAllTimeSlices = True,
    forceSOI = 1
)

import RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi
_phase1_hfrecoNoise = RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi.hfreco.clone(
    inputLabel = "hfprerecoNoise",
    setNoiseFlags = False,
    algorithm = dict(
        Class = "HFSimpleTimeCheck",
        rejectAllFailures = False
    ),
)
_phase1_hfrecoMBNZS = RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi.hfreco.clone(
    inputLabel = "hfprerecoMBNZS",
    setNoiseFlags = False,
    algorithm = dict(
        Class = "HFSimpleTimeCheck",
        rejectAllFailures = False
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
    hbheInput = "hbheprerecoMBNZS"
)
hbheplan1Noise = RecoLocalCalo.HcalRecProducers.hbheplan1_cfi.hbheplan1.clone(
    hbheInput = "hbheprerecoNoise"
)

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( hbherecoMBNZS,
    processQIE11 = True,
# temporarily disabled until RecoLocalCalo/HcalRecProducers/python/HBHEPhase1Reconstructor_cfi.py:flagParametersQIE11 is filled
#    setNoiseFlagsQIE11 = True,
)
run2_HCAL_2017.toModify( hbherecoNoise,
    processQIE11 = True,
# temporarily disabled until RecoLocalCalo/HcalRecProducers/python/HBHEPhase1Reconstructor_cfi.py:flagParametersQIE11 is filled
#    setNoiseFlagsQIE11 = True,
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
