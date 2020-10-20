import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HcalRecProducers.HBHEIsolatedNoiseReflagger_cfi import *
hcalGlobalRecoTask = cms.Task(hbhereco)
hcalGlobalRecoSequence = cms.Sequence(hcalGlobalRecoTask)

from RecoLocalCalo.HcalRecProducers.HBHEPhase1Reconstructor_cfi import hbheprereco as _phase1_hbheprereco

from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toReplaceWith( hbhereco, _phase1_hbheprereco ) # >=Run3

#--- ML-based reco using SONIC+Triton
hbhechannelinfo = _phase1_hbheprereco.clone(
    makeRecHits = False,
    saveInfos = True,
    processQIE8 = False,
)
from RecoLocalCalo.HcalRecProducers.facileHcalReconstructor_cfi import sonic_hbheprereco as _sonic_hbheprereco
from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
(enableSonicTriton & run3_HB).toReplaceWith(hbhereco, _sonic_hbheprereco)
(enableSonicTriton & run3_HB).toModify(hcalGlobalRecoTask, lambda x: x.add(hbhechannelinfo))
