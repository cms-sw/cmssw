import FWCore.ParameterSet.Config as cms

import RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi
hbherecoMB = RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi.hbheprereco.clone(
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

import RecoLocalCalo.HcalRecProducers.hfsimplereco_cfi
hfrecoMB = RecoLocalCalo.HcalRecProducers.hfsimplereco_cfi.hfsimplereco.clone()

import RecoLocalCalo.HcalRecProducers.hosimplereco_cfi
horecoMB = RecoLocalCalo.HcalRecProducers.hosimplereco_cfi.hosimplereco.clone()

# switch off "Hcal ZS in reco":
hfrecoMB.dropZSmarkedPassed = cms.bool(False)
horecoMB.dropZSmarkedPassed = cms.bool(False)

hcalLocalRecoTaskNZS = cms.Task(hbherecoMB,hfrecoMB,horecoMB) 
hcalLocalRecoSequenceNZS = cms.Sequence(hcalLocalRecoTaskNZS) 

import RecoLocalCalo.HcalRecProducers.hfprereco_cfi
import RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi

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

_phase1_hcalLocalRecoTaskNZS = hcalLocalRecoTaskNZS.copy()
_phase1_hcalLocalRecoTaskNZS.add(hfprerecoMB)
 
from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toReplaceWith( hcalLocalRecoTaskNZS, _phase1_hcalLocalRecoTaskNZS )
run2_HF_2017.toReplaceWith( hfrecoMB, _phase1_hfrecoMB )

from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toModify( hbherecoMB,
    processQIE11 = cms.bool(True),
# temporarily disabled until RecoLocalCalo/HcalRecProducers/python/HBHEPhase1Reconstructor_cfi.py:flagParametersQIE11 is filled
#    setNoiseFlagsQIE11 = cms.bool(True),
)

_plan1_hcalLocalRecoTaskNZS = _phase1_hcalLocalRecoTaskNZS.copy()
hbheprerecoMB = hbherecoMB.clone()
_plan1_hcalLocalRecoTaskNZS.add(hbheprerecoMB)
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toReplaceWith(hbherecoMB, hbheplan1MB)
run2_HEPlan1_2017.toReplaceWith(hcalLocalRecoTaskNZS, _plan1_hcalLocalRecoTaskNZS)

hbhecollapseMB = hbheplan1MB.clone()
_collapse_hcalLocalRecoTaskNZS = _phase1_hcalLocalRecoTaskNZS.copy()
_collapse_hcalLocalRecoTaskNZS.add(hbheprerecoMB)
from Configuration.ProcessModifiers.run2_HECollapse_2018_cff import run2_HECollapse_2018
run2_HECollapse_2018.toReplaceWith(hbherecoMB, hbhecollapseMB)
run2_HECollapse_2018.toReplaceWith(hcalLocalRecoTaskNZS, _collapse_hcalLocalRecoTaskNZS)
