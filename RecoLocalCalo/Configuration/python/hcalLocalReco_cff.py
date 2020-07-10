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
hcalLocalRecoTask = cms.Task(hbheprereco,hfreco,horeco,zdcreco)
hcalLocalRecoSequence = cms.Sequence(hcalLocalRecoTask)

from RecoLocalCalo.HcalRecProducers.hfprereco_cfi import hfprereco
from RecoLocalCalo.HcalRecProducers.HFPhase1Reconstructor_cfi import hfreco as _phase1_hfreco
from RecoLocalCalo.HcalRecProducers.hbheplan1_cfi import hbheplan1

#--- for HCALonly wf
hcalOnlyLocalRecoTask = cms.Task(hbheprereco,hfprereco,hfreco,horeco)

# copy for cosmics
_default_hfreco = hfreco.clone()

_phase1_hcalLocalRecoTask = hcalLocalRecoTask.copy()
_phase1_hcalLocalRecoTask.add(hfprereco)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toReplaceWith( hcalLocalRecoTask, _phase1_hcalLocalRecoTask )
run2_HF_2017.toReplaceWith( hfreco, _phase1_hfreco )
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toReplaceWith( hbheprereco, _phase1_hbheprereco )

_plan1_hcalLocalRecoTask = _phase1_hcalLocalRecoTask.copy()
_plan1_hcalLocalRecoTask.add(hbheplan1)
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toReplaceWith(hcalLocalRecoTask, _plan1_hcalLocalRecoTask)

hbhecollapse = hbheplan1.clone()
_collapse_hcalLocalRecoTask = _phase1_hcalLocalRecoTask.copy()
_collapse_hcalLocalRecoTask.add(hbhecollapse)
from Configuration.ProcessModifiers.run2_HECollapse_2018_cff import run2_HECollapse_2018
run2_HECollapse_2018.toReplaceWith(hcalLocalRecoTask, _collapse_hcalLocalRecoTask)

# Run 3 HCAL workflow on GPU
from Configuration.ProcessModifiers.gpu_cff import gpu

from RecoLocalCalo.HcalRecProducers.hbheRecHitProducerGPUTask_cff import *
_hcalLocalRecoTaskGPU = hcalLocalRecoTask.copy()
_hcalLocalRecoTaskGPU.add(hbheRecHitProducerGPUTask)
gpu.toReplaceWith(hcalLocalRecoTask, _hcalLocalRecoTaskGPU)

_hcalOnlyLocalRecoTaskGPU = hcalOnlyLocalRecoTask.copy()
_hcalOnlyLocalRecoTaskGPU.add(hbheRecHitProducerGPUTask)
gpu.toReplaceWith(hcalOnlyLocalRecoTask, _hcalOnlyLocalRecoTaskGPU)

from RecoLocalCalo.HcalRecProducers.hcalCPURecHitsProducer_cfi import hcalCPURecHitsProducer as _hcalCPURecHitsProducer
gpu.toReplaceWith(hbheprereco, _hcalCPURecHitsProducer.clone(
    recHitsM0LabelIn = "hbheRecHitProducerGPU",
    recHitsM0LabelOut = "",
    recHitsLegacyLabelOut = ""
))
#---

_phase2_hcalLocalRecoTask = hcalLocalRecoTask.copy()
_phase2_hcalLocalRecoTask.remove(hbheprereco)

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toReplaceWith( hcalLocalRecoTask, _phase2_hcalLocalRecoTask )


_fastSim_hcalLocalRecoTask = hcalLocalRecoTask.copyAndExclude([zdcreco])
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith( hcalLocalRecoTask, _fastSim_hcalLocalRecoTask )
