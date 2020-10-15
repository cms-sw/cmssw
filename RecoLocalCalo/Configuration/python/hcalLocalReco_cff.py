import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *
from RecoLocalCalo.HcalRecAlgos.hcalChannelPropertiesESProd_cfi import *
hcalOOTPileupESProducer = cms.ESProducer('OOTPileupDBCompatibilityESProducer')

from RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi import hbheprereco as _phase1_hbheprereco
hbheprereco = _phase1_hbheprereco.clone(
    processQIE11 = False,
    tsFromDB = True,
    pulseShapeParametersQIE8 = dict(
        TrianglePeakTS = 4,
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

#--- Phase1 
_phase1_hcalLocalRecoTask = hcalLocalRecoTask.copy()
_phase1_hcalLocalRecoTask.add(hfprereco)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toReplaceWith( hcalLocalRecoTask, _phase1_hcalLocalRecoTask )
run2_HF_2017.toReplaceWith( hfreco, _phase1_hfreco )
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
run2_HCAL_2017.toReplaceWith( hbheprereco, _phase1_hbheprereco )

#--- ML-based reco using SONIC+Triton
from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
hbhechannelinfo = _phase1_hbheprereco.clone(
    makeRecHits = cms.bool(False),
    saveInfos = cms.bool(True),
)
_sonic_hbheprereco = cms.EDProducer("FacileHcalReconstructor",
    Client = cms.PSet(
        batchSize = cms.untracked.uint32(16000),
        address = cms.untracked.string("0.0.0.0"),
        port = cms.untracked.uint32(8001),
        timeout = cms.untracked.uint32(300),
        modelName = cms.string("facile_all_v2"),
        mode = cms.string("Async"),
        modelVersion = cms.string(""),
        verbose = cms.untracked.bool(False),
        allowedTries = cms.untracked.uint32(5),
        outputs = cms.untracked.vstring("output/BiasAdd"),
    ),
    ChannelInfoName = cms.InputTag("hbhechannelinfo")
)
(enableSonicTriton & run2_HCAL_2017).toReplaceWith(hbheprereco, _sonic_hbheprereco)
(enableSonicTriton & run2_HCAL_2017).toModify(hcalLocalRecoTask, lambda x: x.add(hbhechannelinfo))

_plan1_hcalLocalRecoTask = _phase1_hcalLocalRecoTask.copy()
_plan1_hcalLocalRecoTask.add(hbheplan1)
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toReplaceWith(hcalLocalRecoTask, _plan1_hcalLocalRecoTask)

hbhecollapse = hbheplan1.clone()
_collapse_hcalLocalRecoTask = _phase1_hcalLocalRecoTask.copy()
_collapse_hcalLocalRecoTask.add(hbhecollapse)
from Configuration.ProcessModifiers.run2_HECollapse_2018_cff import run2_HECollapse_2018
run2_HECollapse_2018.toReplaceWith(hcalLocalRecoTask, _collapse_hcalLocalRecoTask)

#--- from >=Run3
_run3_hcalLocalRecoTask = _phase1_hcalLocalRecoTask.copy()
_run3_hcalLocalRecoTask.remove(hbheprereco)
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toReplaceWith( hcalLocalRecoTask, _run3_hcalLocalRecoTask )

_fastSim_hcalLocalRecoTask = hcalLocalRecoTask.copyAndExclude([zdcreco])
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith( hcalLocalRecoTask, _fastSim_hcalLocalRecoTask )
